import torch
import logging

from torch import Tensor
import math

from lightning.fabric import Fabric
from hydra.utils import instantiate

from protomotions.agents.utils.data_utils import swap_and_flatten01
from protomotions.utils.replay_buffer import ReplayBuffer
from protomotions.agents.pmp.model import PMPModel
from protomotions.agents.common.common import weight_init
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.ppo.agent import PPO

log = logging.getLogger(__name__)


class PMP(PPO):
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
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        discriminator_optimizers = []
        for i, discriminator in enumerate(model._discriminators):
            optimizer = instantiate(
                self.config.model.config.discriminator_optimizer,
                params=list(discriminator.parameters()),
            )
            discriminator_optimizers.append(optimizer)

        (
            self.model,
            self.actor_optimizer,
            self.critic_optimizer,
            *self.discriminator_optimizers
        ) = self.fabric.setup(
            model,
            actor_optimizer,
            critic_optimizer,
            *discriminator_optimizers
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        super().register_extra_experience_buffer_keys()
        for disc_name in self.model._discriminator_names:
            self.experience_buffer.register_key(f"pmp_rewards_{disc_name}")

    def update_disc_replay_buffer(self, data_dict):
        buf_size = self.pmp_replay_buffer.get_buffer_size()
        buf_total_count = len(self.pmp_replay_buffer)

        values = list(data_dict.values())
        numel = values[0].shape[0]

        for i in range(1, len(values)):
            assert numel == values[i].shape[0]

        if buf_total_count > buf_size:
            keep_probs = (
                torch.ones(numel, device=self.device)
                * self.config.discriminator_replay_keep_prob
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            for k, v in data_dict.items():
                data_dict[k] = v[keep_mask]

        if numel > buf_size:
            rand_idx = torch.randperm(numel)
            rand_idx = rand_idx[:buf_size]
            for k, v in data_dict.items():
                data_dict[k] = v[rand_idx]

        self.pmp_replay_buffer.store(data_dict)

    @torch.no_grad()
    def process_dataset(self, dataset):
        historical_self_obs = swap_and_flatten01(
            self.experience_buffer.historical_self_obs
        )

        num_samples = historical_self_obs.shape[0]

        if len(self.pmp_replay_buffer) == 0:
            replay_historical_self_obs = historical_self_obs
        else:
            replay_dict = self.pmp_replay_buffer.sample(num_samples)
            replay_historical_self_obs = replay_dict["historical_self_obs"]

        expert_historical_self_obs = self.get_expert_historical_self_obs(num_samples)

        discs_number_steps = self.env.self_obs_cb.config.num_historical_steps
        reshaped_historical_self_obs = historical_self_obs.view(
            historical_self_obs.shape[0], discs_number_steps, historical_self_obs.shape[1] // discs_number_steps
        )
        replay_historical_self_obs = replay_historical_self_obs.view(
            replay_historical_self_obs.shape[0], discs_number_steps, replay_historical_self_obs.shape[1] // discs_number_steps
        )
        expert_historical_self_obs = expert_historical_self_obs.view(
            expert_historical_self_obs.shape[0], discs_number_steps, expert_historical_self_obs.shape[1] // discs_number_steps
        )

        # Extract the obs parts based on config indices
        body_part_indices = []
        for i, _ in enumerate(self.model._discriminator_obs_indices):
            body_part_indices.append(
                torch.cat([torch.arange(start, end) for start, end in self.model._discriminator_obs_indices[i]
            ]).to(self.device))

        # Extract parts for each of the discs_number_steps time steps
        agent_historical_self_obs_parts = []
        replay_historical_self_obs_parts = []
        expert_historical_self_obs_parts = []
        discriminator_training_data_dict = {}
        for body_part_id, indices in enumerate(body_part_indices):
            # Process historical observations from the agent
            agent_obs = reshaped_historical_self_obs[..., indices].view(
                historical_self_obs.shape[0], discs_number_steps * indices.numel()
            )
            agent_historical_self_obs_parts.append(agent_obs)
            # Process historical observations from the replay buffer
            replay_obs = replay_historical_self_obs[..., indices].view(
                replay_historical_self_obs.shape[0], discs_number_steps * indices.numel()
            )
            replay_historical_self_obs_parts.append(replay_obs)
            # Process historical observations from demonstration data
            expert_obs = expert_historical_self_obs[..., indices].view(
                expert_historical_self_obs.shape[0], discs_number_steps * indices.numel()
            )
            expert_historical_self_obs_parts.append(expert_obs)

            # Update discriminator training data dictionary
            disc_name = self.model._discriminator_names[body_part_id]
            discriminator_training_data_dict[f"agent_historical_self_obs_{disc_name}"] = agent_obs
            discriminator_training_data_dict[f"replay_historical_self_obs_{disc_name}"] = replay_obs
            discriminator_training_data_dict[f"expert_historical_self_obs_{disc_name}"] = expert_obs

        dataset.update(discriminator_training_data_dict)

        self.update_disc_replay_buffer({"historical_self_obs": historical_self_obs})

        return super().process_dataset(dataset)

    def get_expert_historical_self_obs(self, num_samples: int):
        motion_ids = self.motion_lib.sample_motions(num_samples)
        num_steps = self.env.self_obs_cb.config.num_historical_steps

        dt = self.env.dt
        truncate_time = dt * (num_steps - 1)

        # Since negative times are added to these values in build_historical_self_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip].
        motion_times0 = self.motion_lib.sample_time(
            motion_ids, truncate_time=truncate_time
        )
        motion_times0 = motion_times0 + truncate_time

        obs = self.env.self_obs_cb.build_self_obs_demo(
            motion_ids, motion_times0, num_steps
        ).clone()

        return obs.view(num_samples, -1)

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    @torch.no_grad()
    def calculate_extra_reward(self):
        rew = super().calculate_extra_reward()

        discs_number_steps = self.env.self_obs_cb.config.num_historical_steps
        historical_self_obs = self.experience_buffer.historical_self_obs
        historical_self_obs = historical_self_obs.view(
            self.num_envs * self.num_steps, discs_number_steps, historical_self_obs.shape[-1] // discs_number_steps
        )

        # Extract parts for each discriminator based on specific indices
        historical_self_obs_parts = []
        pmp_rewards = []
        for body_part_id, indices in enumerate(self.model._discriminator_obs_indices):
            part_indices = torch.cat([
                torch.arange(start, end) for start, end in indices
            ]).to(self.device)

            historical_self_obs_part = historical_self_obs[..., part_indices].view(
                self.num_steps * self.num_envs, discs_number_steps * part_indices.numel()
            )
            historical_self_obs_parts.append(historical_self_obs_part)

            pmp_reward = self.model._discriminators[body_part_id].compute_reward(
                {
                    "historical_self_obs": historical_self_obs_part
                }
            ).view(self.num_steps, self.num_envs)
            pmp_rewards.append(pmp_reward)

            self.experience_buffer.batch_update_data(
                f"pmp_rewards_{self.model._discriminator_names[body_part_id]}", pmp_reward
            )

        # Average rewards across all discriminators
        pmp_r = sum(pmp_rewards) / len(pmp_rewards)

        extra_reward = pmp_r * self.config.discriminator_reward_w + rew
        return extra_reward

    # -----------------------------
    # Optimization
    # -----------------------------
    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        extra_opt_steps_dict = super().extra_optimization_steps(batch_dict, batch_idx)
        if batch_idx == 0:
            for discriminator_optimizer in self.discriminator_optimizers:
                discriminator_optimizer.zero_grad()

        if batch_idx < self.discriminator_max_num_batches():
            for body_part_id, body_part_name in enumerate(self.model._discriminator_names):
                discriminator_loss, discriminator_loss_dict = self.discriminator_step(
                    batch_dict,
                    body_part_id=body_part_id,
                    body_part_name=body_part_name
                )
                extra_opt_steps_dict.update(discriminator_loss_dict)
                discriminator_optimizer = self.discriminator_optimizers[body_part_id]
                discriminator_optimizer.zero_grad(set_to_none=True)
                self.fabric.backward(discriminator_loss)
                discriminator_grad_clip_dict = self.handle_model_grad_clipping(
                    self.model._discriminators[body_part_id],
                    discriminator_optimizer,
                    f"discriminator_{body_part_name}",
                )
                extra_opt_steps_dict.update(discriminator_grad_clip_dict)
                discriminator_optimizer.step()

        return extra_opt_steps_dict

    def discriminator_step(
            self,
            batch_dict,
            body_part_id,
            body_part_name
    ):
        agent_obs = batch_dict[f"agent_historical_self_obs_{body_part_name}"][
            : self.config.discriminator_batch_size
        ]
        replay_obs = batch_dict[f"replay_historical_self_obs_{body_part_name}"][
            : self.config.discriminator_batch_size
        ]
        expert_obs = batch_dict[f"expert_historical_self_obs_{body_part_name}"][
            : self.config.discriminator_batch_size
        ]
        combined_obs = torch.cat([agent_obs, expert_obs], dim=0)
        combined_obs.requires_grad_(True)

        combined_dict = self.model._discriminators[body_part_id].compute_logits(
            {"historical_self_obs": combined_obs}, return_norm_obs=True
        )
        combined_logits = combined_dict["outs"]
        combined_norm_obs = combined_dict["norm_historical_self_obs"]

        replay_logits = self.model._discriminators[body_part_id].compute_logits(
            {"historical_self_obs": replay_obs}
        )

        agent_logits = combined_logits[: self.config.discriminator_batch_size]
        expert_logits = combined_logits[self.config.discriminator_batch_size :]

        expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
        unlabeled_loss = torch.nn.functional.softplus(agent_logits).mean()
        replay_loss = torch.nn.functional.softplus(replay_logits).mean()

        neg_loss = 0.5 * (unlabeled_loss + replay_loss)
        class_loss = 0.5 * (expert_loss + neg_loss)

        # Gradient penalty
        disc_grad = torch.autograd.grad(
            combined_logits,
            combined_norm_obs,
            grad_outputs=torch.ones_like(combined_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        disc_grad_norm = torch.sum(torch.square(disc_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_grad_norm)
        grad_loss: Tensor = self.config.discriminator_grad_penalty * disc_grad_penalty

        if self.config.discriminator_weight_decay > 0:
            all_weight_params = self.model._discriminators[body_part_id].all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = total * self.config.discriminator_weight_decay
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.discriminator_logit_weight_decay > 0:
            logit_params = self.model._discriminators[body_part_id].logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])
            logit_weight_decay_loss: Tensor = (
                logit_total * self.config.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        with torch.no_grad():
            pos_acc = self.compute_pos_acc(expert_logits)
            agent_acc = self.compute_neg_acc(agent_logits)
            replay_acc = self.compute_neg_acc(replay_logits)
            neg_acc = 0.5 * (agent_acc + replay_acc)

            log_dict = {
                f"losses/discriminator_loss_{body_part_name}": loss.detach(),
                f"discriminator/pos_acc_{body_part_name}": pos_acc.detach(),
                f"discriminator/agent_acc_{body_part_name}": agent_acc.detach(),
                f"discriminator/replay_acc_{body_part_name}": replay_acc.detach(),
                f"discriminator/neg_acc_{body_part_name}": neg_acc.detach(),
                f"discriminator/grad_penalty_{body_part_name}": disc_grad_penalty.detach(),
                f"discriminator/grad_loss_{body_part_name}": grad_loss.detach(),
                f"discriminator/class_loss_{body_part_name}": class_loss.detach(),
                f"discriminator/l2_logit_total_{body_part_name}": logit_total.detach(),
                f"discriminator/l2_logit_loss_{body_part_name}": logit_weight_decay_loss.detach(),
                f"discriminator/l2_total_{body_part_name}": total.detach(),
                f"discriminator/l2_loss_{body_part_name}": weight_decay_loss.detach(),
                f"discriminator/expert_logit_mean_{body_part_name}": expert_logits.detach().mean(),
                f"discriminator/agent_logit_mean_{body_part_name}": agent_logits.detach().mean(),
                f"discriminator/replay_logit_mean_{body_part_name}": replay_logits.detach().mean(),
            }
            log_dict[f"discriminator/negative_logit_mean_{body_part_name}"] = 0.5 * (
                log_dict[f"discriminator/agent_logit_mean_{body_part_name}"]
                + log_dict[f"discriminator/replay_logit_mean_{body_part_name}"]
            )

        return loss, log_dict

    # -----------------------------
    # Discriminator Metrics and Utility
    # -----------------------------
    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()

    def discriminator_max_num_batches(self):
        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.config.num_discriminator_mini_epochs
            / self.config.batch_size
        )

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def post_epoch_logging(self, training_log_dict):
        for disc_name in self.model._discriminator_names:
            training_log_dict[f"rewards/pmp_rewards_{disc_name}"] = getattr(self.experience_buffer, f"pmp_rewards_{disc_name}").mean()
        super().post_epoch_logging(training_log_dict)
