from roll.agentic.env.base import BaseEnvConfig
from typing import Optional
from omegaconf import OmegaConf, DictConfig # Added import

class DeepResearchEnvConfig(BaseEnvConfig):
    def __init__(
        self,
        max_steps: int = 20,
        reward_step: float = 0.0,
        reward_success: float = 1.0,
        reward_failure: float = -0.5,
        reward_timeout: float = -1.0,
        invalid_act_score: float = 0.0, # From BaseEnvConfig, can be overridden
        task_description_file: Optional[str] = None, # New field
        # Add any environment-specific configurations here, for example:
        # task_description: str = "Default task description",
        # browser_tool_config: dict = None, # Example for browser tool
        # str_editor_workspace_root: str = None, # Example for editor tool
        **kwargs, # Allow additional keyword arguments
    ):
        super().__init__()
        self.max_steps = max_steps
        self.reward_step = reward_step
        self.reward_success = reward_success
        self.reward_failure = reward_failure
        self.reward_timeout = reward_timeout
        self.invalid_act_score = invalid_act_score # Ensure this is set
        self.task_description_file = task_description_file # New field assignment

        # Store other relevant configurations if any
        # self.task_description = task_description
        # self.browser_tool_config = browser_tool_config if browser_tool_config else {}
        # self.str_editor_workspace_root = str_editor_workspace_root

        # Process kwargs to convert OmegaConf DictConfig instances to standard dicts
        processed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, DictConfig):
                processed_kwargs[key] = OmegaConf.to_container(value, resolve=True)
            else:
                processed_kwargs[key] = value
        self.additional_config = processed_kwargs

        # Ensure invalid_act is also part of the config, inherited from BaseEnvConfig
        # self.invalid_act = "" # Default, can be overridden by kwargs if necessary
