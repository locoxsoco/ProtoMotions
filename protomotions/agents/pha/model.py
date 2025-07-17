import torch
from torch import nn
from hydra.utils import instantiate
from protomotions.agents.pmp.model import PMPModel
from protomotions.agents.common.mlp import MultiHeadedMLP


class PHAModel(PMPModel):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

        # Create actors
        self._actors = nn.ModuleList()
        self._actor_names = []
        self._actor_shapes = []
        for actor_config in self.config.actors:
            self._actors.append(instantiate(actor_config))
            self._actor_names.append(actor_config.config.name)
            self._actor_shapes.append(actor_config.config.mu_model.config.num_out)

        print(f"Initialized {len(self._actors)} actors.")
        assert (
                len(self._actors) != 0
        ), "No actors configured. Please provide at least one actor in the config."

        self._critic: MultiHeadedMLP = instantiate(
            self.config.critic,
        )

        # Support for multiple discriminators
        self._discriminators = nn.ModuleList()
        self._discriminator_obs_indices = []
        self._discriminator_names = []
        for disc_config in self.config.discriminators:
            self._discriminators.append(instantiate(disc_config))
            self._discriminator_obs_indices.append(disc_config.config.obs_indices)
            self._discriminator_names.append(disc_config.config.name)

        print(f"Initialized {len(self._discriminators)} discriminators.")
        assert (
                len(self._discriminators) != 0
        ), "No discriminators configured. Please provide at least one discriminator in the config."

    def get_action_and_value(self, input_dict: dict):
        actions = []
        neglogps = []
        for actor_idx, actor in enumerate(self._actors):
            dist = actor(input_dict)
            action = dist.sample()
            logstd = actor.logstd
            std = torch.exp(logstd)
            neglogp = self.neglogp(action, dist.mean, std, logstd)

            actions.append(action)
            neglogps.append(neglogp)

        value = self._critic(input_dict).flatten()

        return actions, neglogps, value.flatten()

    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        actions = []
        for actor in self._actors:
            dist = actor(input_dict)
            if mean:
                action = dist.mean
            else:
                action = dist.sample()

            actions.append(action)

        action = torch.cat(actions, dim=-1)

        return action
