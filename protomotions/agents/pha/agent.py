import torch
import logging

from torch import Tensor
import time

from lightning.fabric import Fabric
from hydra.utils import instantiate
from typing import Tuple, Dict

from protomotions.agents.utils.data_utils import ExperienceBuffer
from protomotions.utils.replay_buffer import ReplayBuffer
from protomotions.agents.pmp.model import PMPModel
from protomotions.agents.common.common import weight_init
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.pmp.agent import PMP
from protomotions.agents.ppo.utils import discount_values, bounds_loss
from rich.progress import track

log = logging.getLogger(__name__)


class PHA(PMP):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        super().__init__(fabric, env, config)
        self.pmp_replay_buffer = ReplayBuffer(self.config.discriminator_replay_size).to(
            self.device
        )

    def setup(self):
        model: PMPModel = instantiate(self.config.model)
        model.apply(weight_init)

        actor_optimizers = []
        for i, actor in enumerate(model._actors):
            actor_optimizer = instantiate(
                self.config.model.config.actor_optimizer,
                params=list(actor.parameters()),
            )
            actor_optimizers.append(actor_optimizer)

        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )

        discriminator_optimizers = []
        for i, discriminator in enumerate(model._discriminators):
            disc_optimizer = instantiate(
                self.config.model.config.discriminator_optimizer,
                params=list(discriminator.parameters()),
            )
            discriminator_optimizers.append(disc_optimizer)

        setup_result = self.fabric.setup(
            model,
            *actor_optimizers,
            critic_optimizer,
            *discriminator_optimizers
        )

        self.model = setup_result[0]
        self.actor_optimizers = setup_result[1:1 + len(actor_optimizers)]
        self.critic_optimizer = setup_result[1 + len(actor_optimizers)]
        self.discriminator_optimizers = setup_result[2 + len(actor_optimizers):]
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    # -----------------------------
    # Optimization
    # -----------------------------
    def optimize_model(self) -> Dict:
        dataset = self.process_dataset(self.experience_buffer.make_dict())
        self.train()
        training_log_dict = {}

        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, training...",
        ):
            iter_log_dict = {}
            dataset_idx = batch_idx % len(dataset)

            # Reshuffle dataset at the beginning of each mini epoch if configured.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Check for NaNs in the batch.
            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")

            # Update actors
            for actor_id, actor_name in enumerate(self.model._actor_names):
                actor_loss, actor_loss_dict = self.actor_step(batch_dict, actor_id, actor_name)
                iter_log_dict.update(actor_loss_dict)
                actor_optimizer = self.actor_optimizers[actor_id]
                actor_optimizer.zero_grad(set_to_none=True)
                self.fabric.backward(actor_loss)
                actor_grad_clip_dict = self.handle_model_grad_clipping(
                    self.model._actors[actor_id], actor_optimizer, f"actor_{actor_name}"
                )
                iter_log_dict.update(actor_grad_clip_dict)
                actor_optimizer.step()
                self.eval()
                with torch.no_grad():
                    dist = self.model._actors[actor_id](batch_dict)
                    logstd = self.model._actors[actor_id].logstd
                    std = torch.exp(logstd)
                    neglogp = self.model.neglogp(batch_dict[f"actions_{actor_name}"], dist.mean, std, logstd)
                    batch_dict[f"new_neglogp_{actor_name}"] = neglogp
                self.train()

            # Update critic
            critic_loss, critic_loss_dict = self.critic_step(batch_dict)
            iter_log_dict.update(critic_loss_dict)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(critic_loss)
            critic_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._critic, self.critic_optimizer, "critic"
            )
            iter_log_dict.update(critic_grad_clip_dict)
            self.critic_optimizer.step()

            # Extra optimization steps if needed.
            extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
            iter_log_dict.update(extra_opt_steps_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()
        return training_log_dict

    def actor_step(
            self,
            batch_dict,
            body_part_id,
            body_part_name
    ) -> Tuple[Tensor, Dict]:
        dist = self.model._actors[body_part_id](batch_dict)
        logstd = self.model._actors[body_part_id].logstd
        std = torch.exp(logstd)
        neglogp = self.model.neglogp(batch_dict[f"actions_{body_part_name}"], dist.mean, std, logstd)

        # Compute probability ratio between new and old policy.
        ratio = torch.exp(batch_dict[f"neglogp_{body_part_name}"] - neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )

        if body_part_id == 0:
           ratio_harl = 1.0
        else:
           ratio_harl = torch.exp(
               torch.sum(
                   torch.stack([
                       batch_dict[f"neglogp_{self.model._actor_names[i]}"] - batch_dict[f"new_neglogp_{self.model._actor_names[i]}"]
                       for i in range(body_part_id)
                   ])
               )
           )

        pha_loss = ratio_harl * torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(dist.mean) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)

        actor_pha_loss = pha_loss.mean()
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_dict, dist)
        actor_loss = actor_pha_loss + b_loss + extra_loss

        log_dict = {
            f"actor/pha_loss_{body_part_name}": actor_pha_loss.detach(),
            f"actor/bounds_loss_{body_part_name}": b_loss.detach(),
            f"actor/extra_loss_{body_part_name}": extra_loss.detach(),
            f"actor/clip_frac_{body_part_name}": clipped.detach(),
            f"losses/actor_loss_{body_part_name}": actor_loss.detach(),
        }

        return actor_loss, log_dict

    # -----------------------------
    # Model Saving and State Dict
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "model": self.model.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }
        for actor_id, actor_optimizer in enumerate(self.actor_optimizers):
            actor_name = self.model._actor_names[actor_id]
            extra_state_dict[f'actor_optimizer_{actor_name}'] = actor_optimizer.state_dict()

        if self.config.normalize_values:
            extra_state_dict["running_val_norm"] = self.running_val_norm.state_dict()
        state_dict.update(extra_state_dict)
        return state_dict

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        for actor_id, actor_optimizer in enumerate(self.actor_optimizers):
            actor_name = self.model._actor_names[actor_id]
            actor_optimizer.load_state_dict(state_dict[f"actor_optimizer_{actor_name}"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    # -----------------------------
    # Experience Buffer and Training Loop
    # -----------------------------
    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        for actor_id, actor_name in enumerate(self.model._actor_names):
            self.experience_buffer.register_key(f"actions_{actor_name}", shape=(self.model._actor_shapes[actor_id],))
            self.experience_buffer.register_key(f"neglogp_{actor_name}")
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                        key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                        range(self.num_steps),
                        description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    actions, neglogps, value = self.model.get_action_and_value(obs)
                    for actor_id, actor_name in enumerate(self.model._actor_names):
                        self.experience_buffer.update_data(f"actions_{actor_name}", step, actions[actor_id])
                        self.experience_buffer.update_data(f"neglogp_{actor_name}", step, neglogps[actor_id])

                    if self.config.normalize_values:
                        value = self.running_val_norm.normalize(value, un_norm=True)
                    self.experience_buffer.update_data("values", step, value)

                    # Check for NaNs in observations and actions
                    for key in obs.keys():
                        if torch.isnan(obs[key]).any():
                            print(f"NaN in {key}: {obs[key]}")
                            raise ValueError("NaN in obs")
                    for action_id, action in enumerate(actions):
                        if torch.isnan(action).any():
                            raise ValueError(f"NaN in action: {self.model._actor_names[action_id]}")

                    action = torch.cat(actions,dim=-1,)
                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(
                            next_value, un_norm=True
                        )
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                    self.config.eval_metrics_every is not None
                    and self.current_epoch > 0
                    and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                            self.best_evaluated_score is None
                            or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)
                training_log_dict.update(eval_log_dict)

            self.post_epoch_logging(training_log_dict)
            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)
