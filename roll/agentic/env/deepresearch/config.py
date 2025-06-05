from roll.agentic.env.base import BaseEnvConfig

class DeepResearchEnvConfig(BaseEnvConfig):
    def __init__(
        self,
        max_steps: int = 20,
        reward_step: float = 0.0,
        reward_success: float = 1.0,
        reward_failure: float = -0.5,
        reward_timeout: float = -1.0,
        invalid_act_score: float = 0.0, # From BaseEnvConfig, can be overridden
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

        # Store other relevant configurations if any
        # self.task_description = task_description
        # self.browser_tool_config = browser_tool_config if browser_tool_config else {}
        # self.str_editor_workspace_root = str_editor_workspace_root

        # Store any additional kwargs as attributes, if needed, or pass to a dict
        self.additional_config = kwargs

        # Ensure invalid_act is also part of the config, inherited from BaseEnvConfig
        # self.invalid_act = "" # Default, can be overridden by kwargs if necessary
