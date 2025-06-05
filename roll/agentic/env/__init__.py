"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN
"""

# from .alfworld.config import AlfredEnvConfig
# from .alfworld.env import AlfredTXTEnv
# from .bandit.config import BanditEnvConfig
# from .bandit.env import BanditEnv
# from .countdown.config import CountdownEnvConfig
# from .countdown.env import CountdownEnv
from .sokoban.config import SokobanEnvConfig
from .sokoban.env import SokobanEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .frozen_lake.env import FrozenLakeEnv
from .deepresearch import DeepResearchEnv, DeepResearchTask, DeepResearchEnvConfig # Added
# from .metamathqa.env import MetaMathQAEnv
# from .metamathqa.config import MetaMathQAEnvConfig


REGISTERED_ENVS = {
    # "bandit": BanditEnv,
    # "countdown": CountdownEnv,
    "sokoban": SokobanEnv,
    "frozen_lake": FrozenLakeEnv,
    DeepResearchTask.env_name: DeepResearchEnv, # New line
    # 'alfworld': AlfredTXTEnv,
    # "metamathqa": MetaMathQAEnv,
}

REGISTERED_ENV_CONFIGS = {
    # "bandit": BanditEnvConfig,
    # "countdown": CountdownEnvConfig,
    "sokoban": SokobanEnvConfig,
    "frozen_lake": FrozenLakeEnvConfig,
    DeepResearchTask.env_name: DeepResearchEnvConfig, # New line
    # 'alfworld': AlfredEnvConfig,
    # "metamathqa": MetaMathQAEnvConfig,
}

try:
    from .webshop.env import WebShopEnv
    from .webshop.config import WebShopEnvConfig

    REGISTERED_ENVS["webshop"] = WebShopEnv
    REGISTERED_ENV_CONFIGS["webshop"] = WebShopEnvConfig
except Exception as e:
    pass
