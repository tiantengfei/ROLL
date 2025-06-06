from dataclasses import dataclass # Added
from typing import Optional # Keep Optional for task_description_file
# Removed: from omegaconf import OmegaConf, DictConfig (no longer needed)
from roll.agentic.env.base import BaseEnvConfig

@dataclass
class DeepResearchEnvConfig(BaseEnvConfig):
    # Fields from BaseEnvConfig (like invalid_act_score, invalid_act)
    # will be initialized by super().__init__() in __post_init__.
    # DeepResearchEnvConfig specific fields:
    max_steps: int = 20
    reward_step: float = 0.0
    reward_success: float = 1.0
    reward_failure: float = -0.5
    reward_timeout: float = -1.0
    # invalid_act_score is initialized in BaseEnvConfig. If DeepResearchEnvConfig
    # needs a *different* default than BaseEnvConfig, it should be declared here.
    # Since the old __init__ had `invalid_act_score: float = 0.0`, which matches
    # BaseEnvConfig's default setting, we don't need to re-declare it here.
    task_description_file: Optional[str] = None

    # The 'additional_config' field and **kwargs processing are removed.
    # If unknown fields are provided via YAML, Hydra/OmegaConf might still allow them
    # at instantiation time if strict mode is not enabled for the dataclass,
    # but they won't be part of the dataclass fields themselves.
    # For asdict() to work correctly with only defined fields, this is the desired state.

    def __post_init__(self):
        super().__init__()
        # If any specific logic was needed after BaseEnvConfig.__init__ and after
        # dataclass fields are set, it would go here.
        # For example, if `invalid_act_score` specific to DeepResearch needed
        # to be different from BaseEnvConfig's default and wasn't overridden by YAML:
        # if self.invalid_act_score == 0.0: # Default from BaseEnvConfig
        #     self.invalid_act_score = -0.1 # Example: DeepResearch specific default
        # However, based on the previous __init__, it used the same default as BaseEnvConfig.
        pass
