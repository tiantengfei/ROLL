# Copyright (c) 2025, ALIBABA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any

from roll.configs.base_config import BaseConfig, ScheduleConfig
from roll.configs.worker_config import WorkerConfig
from roll.utils.logging import get_logger

logger = get_logger()


@dataclass
class DatasetFilterConfig:
    source: Optional[str] = None
    min_difficulty: Optional[float] = None
    max_difficulty: Optional[float] = None
    num_samples: int = 0

@dataclass
class RewardFilterConfig:
    type: Literal["no_filter", "mean_filter", "std_filter"] = field(
        default="no_filter",
        metadata={"help": "Type of filter to apply to rewards."},
    )
    filter_args: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Arguments used in `filter_fn`"},
    )


@dataclass
class RewardConfig(WorkerConfig):
    code_url: str = field(
        default=None,
        metadata={"help": "The url of the code."}
    )
    use_local: bool = field(
        default=True,
        metadata={"help": "Whether to use local code instead of downloading from URL."}
    )
    judge_prompt: str = field(
        default=None,
        metadata={"help": "The prompt for judge."}
    )
    judge_model_type: str = field(
        default=None,
        metadata={"help": "api or inference"}
    )
    judge_model_name: str = field(
        default=None,
        metadata={"help": "judge_model_name."}
    )
    judge_api_url: str = field(
        default=None,
        metadata={"help": "judge_api_url."}
    )
    judge_api_key: str = field(
        default=None,
        metadata={"help": "judge_api_key."}
    )
    format_pattern: str = field(
        default=None,
        metadata={"help": "The pattern of the answer format."}
    )
    reward_type: str = field(default=None, metadata={"help": "The type of the reward."})
    response_length_penalty_coef: float = field(default=0.0, metadata={"help": "The coefficient of the response length penalty."})
    
    tag_included: List[str] = field(default_factory=list, metadata={"help": "The tags of the domain."})
    query_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward query filtering"},)
    response_filter_config: RewardFilterConfig = field(
        default_factory=RewardFilterConfig,
        metadata={"help": "Arguments passed to reward response filtering"},
    )



@dataclass
class RLVRConfig(BaseConfig):
    # global
    global_template: str = field(
        default=None,
        metadata={"help": "The template of the global."})
    dataset_filter: DatasetFilterConfig = field(
        default_factory=DatasetFilterConfig,
        metadata={"help": "Configuration for filtering dataset by source and difficulty"},
    )
    num_return_sequences_in_group: int = field(
        default=1,
        metadata={"help": "The number of return sequences in one group, used in generation_args."}
    )

    generate_opt_level: int = field(
        default=1,
        metadata={
            "help": "generate optimizing level: 0 use base batch generate interface, 1 use scheduler process requests"
        },
    )
    is_num_return_sequences_expand: bool = field(
        default=False,
        metadata={"help": "whether replicate `num_return_sequences` times in prompts or not."}
    )
    max_running_requests: int = field(
        default=128,
        metadata={"help": "The maximum number of running requests."}
    )
    is_use_additional_prompts: bool = field(
        default=False,
        metadata={"help": "Whether to use additional prompts or not."}
    )
    max_additional_running_prompts: int = field(
        default=16, metadata={"help": "The additional number of running prompts, beyond batch_size."}
    )

    # role related
    pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory, if available."})
    reward_pretrain: str = field(
        default=None,
        metadata={"help": "Path to pretrain model directory for the reward model, if available."}
    )
    validation: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the validation."}
    )
    actor_train: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's training role."}
    )
    actor_infer: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the actor's inference role."}
    )
    critic: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the critic's training role."}
    )
    reference: WorkerConfig = field(
        default_factory=WorkerConfig,
        metadata={"help": "Configuration for the reference role."}
    )
    rewards: Optional[Dict[str, RewardConfig]] = field(
        default_factory=dict,
        metadata={"help": "Configuration for the multi domain rewards."}
    )

    # PPO related
    ppo_epochs: int = field(default=1, metadata={"help": "Number of optimisation epochs per batch of samples"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Maximum norm"})
    l2: float = field(default=0.0, metadata={"help": "L2 regularization"})
    lambd: float = field(default=0.95, metadata={"help": "Gamma parameter for advantage calculation"})
    gamma: float = field(default=1, metadata={"help": "Lambda parameter for advantage calculation"})
    pg_clip: Optional[float] = field(default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"})
    value_clip: Optional[float] = field(
        default=None, metadata={"help": "Range for clipping values in loss calculation"}
    )
    kl_penalty: Literal["kl", "abs", "mse", "full"] = field(
        default="kl",
        metadata={
            "help": "kl penalty options: 'kl': model_logp - ref_logp, 'abs': abs(kl), 'mse': "
                    "mean squared error mse(kl) and 'full': the actual kl for all tokens in the distribution"
        },
    )
    target_kl: Optional[float] = field(default=None, metadata={"help": "Target KL value for adaptive KL control"})
    init_kl_coef: float = field(
        default=0.2, metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"}
    )
    kl_horizon: int = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    use_reward_scaling: bool = field(default=False, metadata={"help": "Use reward scaling"})
    add_len_reward: bool = field(default=False)
    reward_clip: float = field(default=None, metadata={"help": "reward clip value."})
    difficulty_loss_weight: bool = field(default=False, metadata={"help": "Use difficulty_loss_weight"})
    length_loss_weight: bool = field(default=False, metadata={"help": "Use length_loss_weight"})
    use_reward_norm: bool = field(
        default=False, metadata={"help": "Use reward normalization. Only applicable if use_reward_scaling is True."}
    )
    whiten_rewards: bool = field(default=False, metadata={"help": "Whiten the rewards before compute advantages."})
    whiten_advantages: bool = field(default=False, metadata={"help": "Whiten the advantage."})
    advantage_clip: float = field(default=None, metadata={"help": "advantage_clip value"})
    adv_estimator: Literal["gae", "reinforce", "grpo"] = field(
        default="gae", metadata={"help": "advantage estimator: gae (GAE)."}
    )
    reward_norm: Literal["batch", "group", "running", None] = field(
        default=None,
        metadata={
            "help": "Reward normalization type: 'batch' (normalize across batch), 'group' (normalize within prompt groups), 'running' (use running statistics)"
        },
    )
    reward_shift: bool = field(
        default=False, metadata={"help": "Only subtract mean without dividing by std during reward normalization"}
    )
    reward_scale: bool = field(
        default=False, metadata={"help": "Only divide by std without subtracting mean during reward normalization"}
    )
    add_token_level_kl: bool = field(default=False, metadata={"help": "Add token level kl penalty"})
    critic_warmup: int = field(
        default=0,
        metadata={"help": "Pre-training step for critic model"},
    )
    use_kl_loss: bool = field(default=False, metadata={"help": "Use kl loss"})
    kl_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for kl loss"})
    entropy_loss_coef: float = field(default=0, metadata={"help": "Loss coefficient for entropy loss"})
    sft_loss_coef: float = field(
        default=0,
        metadata={"help": "Loss coefficient for SFT loss, used for positive samples"}
    )
    use_topr_loss: bool = field(
        default=False,
        metadata={"help": "whether to use TPRO loss, http://arxiv.org/abs/2503.14286"}
    )
    rl_loss_coef: float = field(
        default=1.0,
        metadata={"help": "Loss coefficient for RL loss"}
    )
    dual_clip_loss: bool = field(default=False, metadata={"help": "Use dual clip loss"})
    loss_agg_mode: Literal["token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "seq-mean-token-sum-norm"] = (
        field(default="token-mean", metadata={"help": "Loss aggregation mode"})
    )
    val_greedy: bool = field(default=False, metadata={"help": "Use greedy for validation"})
    val_n_sample: int = field(default=1, metadata={"help": "Number of samples for validation"})
    max_len_mask: bool = field(default=False)
    mask_type: Literal["all", "loss"] = field(default="loss", metadata={"help": "Mask type: 'all' or 'loss'"})
    difficulty_mask: bool = field(default=False)
    balance_length: bool = field(default=False)
    minibatch_data_iter_num: int = field(default=1)
    difficulty_low_threshold: float = field(default=0.0)
    difficulty_high_threshold: float = field(default=1.0)
    error_max_len_clip: bool = field(default=False)
    error_max_len_threshold: int = field(default=9999999999)

    def __post_init__(self):
        super().__post_init__()

        if (
                self.actor_train.model_args.model_name_or_path is None
                or self.actor_infer.model_args.model_name_or_path
                or self.reference.model_args.model_name_or_path is None
        ):
            self.actor_train.model_args.model_name_or_path = self.pretrain
            self.actor_infer.model_args.model_name_or_path = self.pretrain
            self.reference.model_args.model_name_or_path = self.pretrain

        # default worker_cls
        if self.actor_train.worker_cls is None:
            self.actor_train.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.actor_infer.worker_cls is None:
            self.actor_infer.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.reference.worker_cls is None:
            self.reference.worker_cls = "roll.pipeline.rlvr.actor_worker.ActorWorker"
        if self.critic.worker_cls is None:
            self.critic.worker_cls = "roll.pipeline.base_worker.CriticWorker"

        if self.critic.model_args.model_name_or_path is None:
            self.critic.model_args.model_name_or_path = self.reward_pretrain

        self.actor_train.training_args.output_dir = self.output_dir
        self.actor_infer.training_args.output_dir = self.output_dir
        self.critic.training_args.output_dir = self.output_dir

        self.actor_infer.generating_args.num_return_sequences = self.num_return_sequences_in_group

        self.actor_infer.name = "actor_infer"
        self.actor_train.name = "actor_train"
        self.reference.name = "reference"
        self.critic.name = "critic"
        self.domain_2_tag = None
        self.tag_2_domain = None
        if self.rewards is not None:
            self.domain_2_tag = {key: set(worker_config.tag_included) for key, worker_config in self.rewards.items()}
            self.tag_2_domain = {
                tag: key for key, worker_config in self.rewards.items() for tag in worker_config.tag_included
            }

    def set_max_steps(self, max_steps: int):
        actor_backward_batch_size = (
                self.actor_train.training_args.per_device_train_batch_size
                * self.actor_train.training_args.gradient_accumulation_steps
        )
        critic_backward_batch_size = (
                self.critic.training_args.per_device_train_batch_size
                * self.critic.training_args.gradient_accumulation_steps
        )
        # Not divided by dp_size yet, need to divide after distributed environment initialization
        self.actor_train.training_args.max_steps = max_steps * (
                self.rollout_batch_size
                * self.actor_infer.generating_args.num_return_sequences
                * self.ppo_epochs
                // actor_backward_batch_size
        )
        self.critic.training_args.max_steps = max_steps * (
                self.rollout_batch_size
                * self.actor_infer.generating_args.num_return_sequences
                // critic_backward_batch_size
        )

        logger.info(f"pipeline max_steps: {self.max_steps} to {max_steps}")
        logger.info(f"actor train max_steps without dp_size: {self.actor_train.training_args.max_steps}")
        logger.info(f"critic train max_steps without dp_size: {self.critic.training_args.max_steps}")
        self.max_steps = max_steps

    def to_dict(self):
        return dataclasses.asdict(self)
