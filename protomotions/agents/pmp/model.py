from torch import nn
from hydra.utils import instantiate
from protomotions.agents.ppo.model import PPOModel


class PMPModel(PPOModel):
    def __init__(self, config):
        super().__init__(config)
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
