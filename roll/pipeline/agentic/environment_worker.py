import copy
import re
import json
import os
import logging
import random # Added import
from dataclasses import dataclass, field, asdict
from itertools import zip_longest
from threading import Thread
from typing import Dict, List, Optional, Union, Tuple, Any

import PIL
import numpy as np
import ray
import torch
from ray.util.queue import Queue
from tensordict import TensorDict
from transformers import AutoTokenizer

from roll.agentic.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.generate_scheduler import OneRequestScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.agentic.agentic_config import EnvManagerConfig
from roll.utils.functionals import pad_to_length


"""
base agentic codes reference: https://github.com/RAGEN-AI/RAGEN/blob/main/ragen/llm_agent/es_manager.py
"""

@dataclass
class EnvStatus:
    """Status of an environment"""

    truncated: bool = False  # done but not success
    terminated: bool = False  # done and success
    num_actions: int = 0  # current action step (single action)
    rewards: List[float] = field(default_factory=list)  # rewards for each turn
    seed: Optional[int] = None  # what seed is used to reset this environment
    step: int = 0  # current step (single step)
    initial_system_prompt: Optional[str] = None # Added for DeepResearchEnv
    task_description: Optional[str] = None      # Added for DeepResearchEnv

    @property
    def done(self):
        return self.truncated or self.terminated


def get_masks_and_scores(
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    all_scores: List[List[float]] = None,
    use_turn_scores: bool = False,
):
    """
    input_ids: shape (bsz, seq_len)
    all_scores: list[list[float], 存储每个env每轮的reward
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    NOTE: important! input_ids is left pad
    """
    assistant_turn_start_tokens = tokenizer.encode("<|im_start|>assistant\n")
    turn_start_token = assistant_turn_start_tokens[0]
    turn_starts = torch.where(input_ids == turn_start_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)

    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)  # only learns all assistant turns
    non_prompt_mask = turn_indicators > 2  # learns everything after system prompt + user prompts

    # turn text: '<|im_start|>assistant\n<answer>Right</answer><|im_end|>'
    # <|im_start|>assistant\n 应该mask掉才对，保留<|im_end|>
    for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
        turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
        turn_start_position = (input_ids == turn_start_token) & (turn_indicators == turn_indicator)
        batch_size, seq_len = input_ids.shape
        num_tokens = len(assistant_turn_start_tokens)
        turn_start_indices = turn_start_position.nonzero(as_tuple=True)
        mask_matrix = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        for batch_idx, start_idx in zip(turn_start_indices[0], turn_start_indices[1]):
            end_idx = start_idx + num_tokens
            if end_idx <= seq_len:
                mask_matrix[batch_idx, start_idx:end_idx] = True
        response_mask[mask_matrix] = False
        if idx == 0:
            non_prompt_mask[mask_matrix] = False

    reward_token = tokenizer.encode("<|im_end|>")[0]
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3  # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)

    return non_prompt_mask, score_tensor, response_mask


def left_pad_2_right(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    non_prompt_mask: torch.Tensor,
    pad_token_id: int,
    score_tensor: torch.Tensor,
):
    """
    Convert left-padded tensors to right-padded tensors.
    """
    batch_size = input_ids.size(0)
    first_one = attention_mask.float().argmax(dim=1)

    for i in range(batch_size):
        shift = first_one[i].item()
        if shift > 0:
            input_ids[i, :-shift] = input_ids[i, shift:].clone()
            input_ids[i, -shift:] = pad_token_id
            attention_mask[i, :-shift] = attention_mask[i, shift:].clone()
            attention_mask[i, -shift:] = False
            response_mask[i, :-shift] = response_mask[i, shift:].clone()
            response_mask[i, -shift:] = False
            non_prompt_mask[i, :-shift] = non_prompt_mask[i, shift:].clone()
            non_prompt_mask[i, -shift:] = False
            score_tensor[i, :-shift] = score_tensor[i, shift:].clone()
            score_tensor[i, -shift:] = 0


class EnvironmentWorker(Worker):
    """
    1. 一个EnvironmentWorker(进程)持有一个env实例: 执行env.reset, env.step, 管理rollout的状态
        group trajectory表达: group内的init state一致，依赖env_config 中的seed来控制, 一个group内env 对应episode的seed一致
        不采用持有envs的原因是，envs需要管理一组env的交互，增加描述的复杂性
    2. 持有infer_cluster ref, 用于async generate
    3. run_rollout_loop, 持续rollout trajectory, 将done的trajectory回传到output_queue

    承担EnvStateManager的history收集功能
    一个group内的env reset进度应该一致

    TODO: env并行方式后续改成进程+线程并行：目的解决一个env占用一个进程对系统资源的开销
          - 一个EnvironmentWorker持有n个EnvStateManager
          - EnvStateManager管理一个env的rollout loop
          - EnvStateManager.run_rollout_loop,运行在n个线程里
    TODO: GiGPO: https://arxiv.org/abs/2505.10978
    """

    def __init__(self, worker_config: EnvManagerConfig):
        super().__init__(worker_config)
        self.worker_config: EnvManagerConfig = worker_config
        self.env_config: Dict = worker_config.env_configs[self.rank]
        self.env_entry = None
        self.output_queue = None
        self.input_queue = None
        self.infer_worker = None
        self.rollout_cache = None
        self.mode = "train"
        self.group_seed = None
        self.episode_id = 0
        self.process_input_queue_thread = None
        self.running = False
        self.generate_scheduler = None

        self.prefix_lookup = None
        self.env_config_lookup = None
        self.tokenizer = None
        self.deepresearch_tasks_by_split: Dict[str, Dict[str, List[str]]] = {}

    def initialize(self, pipeline_config, infer_worker, input_queue: Queue, output_queue: Queue, mode: str = "train"):
        super().initialize(pipeline_config)
        self.output_queue = output_queue
        self.input_queue = input_queue
        self.infer_worker = infer_worker
        self.rollout_cache = None
        self.mode = mode

        self.env_entry = copy.deepcopy(self.env_config)
        self.env_entry["env"] = REGISTERED_ENVS[self.env_entry["env_class"]](self.env_entry["config"])
        self.env_entry["status"] = EnvStatus()

        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

        self._init_prefix_lookup()
        self.generate_scheduler = OneRequestScheduler.remote(
            infer_worker=self.infer_worker, pipeline_config=self.pipeline_config
        )

    def reset(self):
        entry = self.env_entry
        self.rollout_cache = {
            "env_id": entry["env_id"],
            "history": [],
            "group_id": entry["group_id"],
            "tag": entry["tag"],
            "penalty": 0,
            "frames": [],
        }

        seed = self.group_seed + self.episode_id # This is the episode-specific seed

        current_task_description = "" # Initialize

        is_deepresearch_env = (entry.get("tag") == "DeepResearchEnv")

        if is_deepresearch_env:
            env_tag = entry.get("tag") # ensure env_tag is defined
            tasks_for_tag = self.deepresearch_tasks_by_split.get(env_tag, {"train": [], "validation": []})
            train_task_list = tasks_for_tag.get("train", [])
            validation_task_list = tasks_for_tag.get("validation", [])

            relevant_task_list = []
            selected_task_type_log = ""

            if self.mode == "train":
                relevant_task_list = train_task_list
                selected_task_type_log = "train"
                if not relevant_task_list:
                    self.logger.warning(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) no 'train' tasks. Fallback to 'validation'.")
                    relevant_task_list = validation_task_list
                    selected_task_type_log = "train (fallback to validation)"
            elif self.mode == "val" or self.mode == "eval":
                relevant_task_list = validation_task_list
                selected_task_type_log = self.mode
                if not relevant_task_list:
                    self.logger.warning(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) no '{selected_task_type_log}' tasks. Fallback to 'train'.")
                    relevant_task_list = train_task_list
                    selected_task_type_log = f"{self.mode} (fallback to train)"
            else:
                self.logger.warning(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}) unknown mode '{self.mode}'. Defaulting to 'train' tasks.")
                relevant_task_list = train_task_list
                selected_task_type_log = "unknown mode (defaulting to train)"
                if not relevant_task_list:
                    relevant_task_list = validation_task_list
                    selected_task_type_log = "unknown mode (fallback to validation)"

            if relevant_task_list:
                if self.mode == "train":
                    task_rng = random.Random(seed)
                    current_task_description = task_rng.choice(relevant_task_list)
                    self.logger.info(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) random task from '{selected_task_type_log}': {current_task_description[:100]}...")
                else: # val or eval
                    if len(relevant_task_list) > 0:
                        task_index = self.episode_id % len(relevant_task_list)
                        current_task_description = relevant_task_list[task_index]
                        self.logger.info(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) sequential task {task_index + 1}/{len(relevant_task_list)} from '{selected_task_type_log}': {current_task_description[:100]}...")
                    else: # Should not be reached if relevant_task_list is not empty, but defensive.
                        current_task_description = self.prefix_lookup.get(entry["env_id"], f"Fallback for DeepResearchEnv {env_tag} - {selected_task_type_log} list was empty after check.")
                        self.logger.warning(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) using fallback prefix_lookup for {selected_task_type_log}: {current_task_description[:100]}...")
            else: # No tasks in relevant_task_list (primary or fallback)
                current_task_description = self.prefix_lookup.get(entry["env_id"], f"Fallback for DeepResearchEnv {env_tag} - all task lists empty for mode {self.mode}.")
                self.logger.warning(f"DeepResearchEnv (tag: {env_tag}, env_id: {entry['env_id']}, mode: {self.mode}) using fallback prefix_lookup (all lists empty): {current_task_description[:100]}...")
        else:
            # For other environment types, use the prefix_lookup as before
            current_task_description = self.prefix_lookup.get(entry["env_id"], "Default task for non-DeepResearch env.")

        reset_output = entry["env"].reset(seed=seed, task_description=current_task_description)

        current_env_status = EnvStatus(seed=seed)

        if entry["tag"] == "DeepResearchEnv": # Check for DeepResearchEnv
            # Store the specific parts from DeepResearchEnv's reset output
            current_env_status.initial_system_prompt = reset_output.get("initial_observation")
            current_env_status.task_description = current_task_description
            next_state = current_task_description
        else:
            # Existing logic for other environments
            next_state = self._handle_mm_state(entry["env"].render())

        entry["status"] = current_env_status

        # update rollout cache
        self.rollout_cache["history"] = self._update_cache_history(
            self.rollout_cache["history"],
            next_state=next_state, # This will be task_description for DeepResearchEnv
            actions_left=entry["max_actions_per_traj"],
            num_actions_info=None,
        )
        self.episode_id += 1
        return self.rollout_cache

    def step(self, llm_output: DataProto):
        env_input: Dict = self.get_env_input(llm_output)

        entry = self.env_entry
        actions_left_before = entry["max_actions_per_traj"] - entry["status"].num_actions

        # execute actions in env
        valid_actions = self._extract_map_valid_actions(entry, env_input["actions"])

        acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(
            entry["env"], valid_actions[:actions_left_before]
        )

        if len(valid_actions) != len(env_input["actions"]) or not valid_actions:
            self.rollout_cache["penalty"] += self.worker_config.format_penalty

        status, history = self._log_env_state(
            entry["status"],
            self.rollout_cache["history"],
            entry["env"].render(),
            entry["max_actions_per_traj"],
            executed_actions,
            valid_actions,
            acc_reward,
            turn_done,
            turn_info,
            env_input,
        )
        status.step += 1
        entry["status"] = status

        max_steps_per_traj = entry.get("max_steps_per_traj", entry["max_actions_per_traj"])
        if status.step >= max_steps_per_traj and not turn_done:
            entry["status"].truncated = True
            entry["status"].terminated = True

        self.rollout_cache["history"] = history

        frame = entry["env"].render(mode="rgb_array")
        if isinstance(frame, np.ndarray):
            self.rollout_cache["frames"].append(frame)

        return status

    def generate(self, env_output: Dict):
        lm_input: DataProto = self.get_lm_input(env_output, prepare_for_update=False)
        lm_input.meta_info = env_output["meta_info"]

        generation_config = self.worker_config.generating_args.to_dict()
        generation_config["max_new_tokens"] = min(
            generation_config["max_new_tokens"],
            max(self.pipeline_config.sequence_length - lm_input.batch["input_ids"].shape[1] - 1, 1),
        )
        if generation_config["max_new_tokens"] <= 1:
            self.logger.warning(
                f"sequence_length = {self.pipeline_config.sequence_length} input_ids length = {lm_input.batch['input_ids'].shape[1]},"
                f"maybe you should increase the response_length"
            )
            return None

        gen_batch = lm_input.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
        gen_batch.meta_info["generation_config"] = generation_config
        gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
        lm_output: DataProto = ray.get(self.generate_scheduler.generate_one_request.remote(data=gen_batch))

        if lm_output is not None:
            # 未被abort
            gen_batch.meta_info.pop("generation_config")
            gen_batch.meta_info.pop("response_callback_fn")
            lm_input = lm_input.repeat(repeat_times=generation_config["num_return_sequences"])
            lm_output.union(lm_input)
        return lm_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_rollout_loop(self, data: DataProto):
        """
        1. 每次调用run_rollout_loop,
            会持续的play episode, 直到收到采集完成的command
            需要重置seed, 确保每个group的seed一致
            episode_id 置0
        seed更新逻辑:
            group_seed = seed + group_seed
            episode_seed = group_seed + episode_id

        trajectory_id: f"{group_id}_{episode_id}_{episode_seed}"
        """

        self.start_input_queue_process()
        self.running = True
        self.episode_id = 0

        self.group_seed = data.meta_info["seed"] + self.env_entry["group_seed"]
        env_output = self.reset()
        env_output["meta_info"] = data.meta_info

        while self.running:
            lm_output: DataProto = self.generate(env_output)

            status = EnvStatus(truncated=True, terminated=True)
            if lm_output is not None:
                status: EnvStatus = self.step(lm_output)

            if status.done and self.running:
                rollout: DataProto = self.formulate_rollouts()
                traj_group_id = f"{self.env_entry['group_id']}_{self.episode_id}_{self.group_seed}"
                rollout.non_tensor_batch["traj_group_id"] = np.array([traj_group_id], dtype=object)
                self.output_queue.put(rollout)
                self.rollout_cache = None
                if self.episode_id >= self.worker_config.max_traj_per_env:
                    self.logger.debug(
                        f"max_traj_per_env {self.worker_config.max_traj_per_env} reached, stopping rollout loop"
                    )
                    break
                self.reset()

        self.process_input_queue_thread.join()

    def get_lm_input(self, env_output, prepare_for_update: bool) -> DataProto:
        """"""
        llm_input_texts, messages_list = self._format_messages(
            env_output=env_output, prepare_for_update=prepare_for_update, use_raw_llm_response=False
        )
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )
        # (bsz, seq_len), bsz=1
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=input_ids.shape[0],
        )
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"]], dtype=object),
            "group_ids": np.array([env_output["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([env_output["tag"]], dtype=object),
        }
        return llm_inputs

    def get_env_input(self, lm_output: DataProto) -> Dict:
        if lm_output.batch is not None and "responses" in lm_output.batch.keys():
            responses = self.tokenizer.batch_decode(lm_output.batch["responses"], skip_special_tokens=True)
        else:  # dataproto has textual responses
            responses = lm_output.non_tensor_batch["response_texts"]
        responses = [
            "<think>" + response if self.pipeline_config.enable_think else "<answer>" + response
            for response in responses
        ]  # The LLM generation does not include <think> tags. Add them back here.

        env_ids = lm_output.non_tensor_batch["env_ids"]
        env_id = env_ids[0]
        response = responses[0]
        llm_response, actions = self._parse_response(response)
        env_input = {
            "env_id": env_id,
            "llm_raw_response": response,
            "llm_response": llm_response,
            "actions": actions,
        }
        return env_input

    def formulate_rollouts(self):
        """
        1. 每个env的trajectory 是一个rollout
        2. 每个rollout 是一个List[Dict]
        3. 每个Dict 是一个step的信息
        """
        llm_input_texts, messages_list = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=False
        )
        inputs = self.tokenizer(
            llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False
        )
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        scores = [[i["reward"] for i in self.rollout_cache["history"]]]
        episode_scores = [sum(i) for i in scores]
        penalty = self.rollout_cache["penalty"]

        non_prompt_mask, score_tensor, response_mask = get_masks_and_scores(
            input_ids, self.tokenizer, scores, use_turn_scores=self.pipeline_config.use_turn_scores
        )
        non_prompt_mask = torch.logical_and(non_prompt_mask, attention_mask)
        response_mask = torch.logical_and(response_mask, attention_mask)

        left_pad_2_right(
            input_ids, attention_mask, response_mask, non_prompt_mask, self.tokenizer.pad_token_id, score_tensor
        )

        response_length = response_mask.sum(dim=-1).float().mean().item()
        input_ids = pad_to_length(
            input_ids, length=self.pipeline_config.sequence_length, pad_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_to_length(attention_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        response_mask = pad_to_length(response_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        non_prompt_mask = pad_to_length(non_prompt_mask, length=self.pipeline_config.sequence_length, pad_value=0)
        score_tensor = pad_to_length(score_tensor, length=self.pipeline_config.sequence_length, pad_value=0)

        position_ids = attention_mask.cumsum(dim=-1)
        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "penalty": torch.Tensor([penalty]),
            },
            batch_size=input_ids.shape[0],
        )
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([self.rollout_cache["env_id"]], dtype=object),
            "group_ids": np.array([self.rollout_cache["group_id"]], dtype=object),
            "messages_list": np.array(messages_list, dtype=object),
            "tags": np.array([self.rollout_cache["tag"]], dtype=object),
            "frames": np.array([self.rollout_cache["frames"]], dtype=object),
        }
        # pad to response length
        llm_inputs.batch["llm_response_mask"] = response_mask
        llm_inputs.batch["non_prompt_mask"] = non_prompt_mask
        llm_inputs.batch["response_mask"] = non_prompt_mask
        if self.pipeline_config.enable_response_mask:
            # 只使用llm的response mask，不包含环境的state
            llm_inputs.batch["response_mask"] = response_mask
        first_true_indices = non_prompt_mask.int().argmax(dim=1)
        no_true_mask = ~non_prompt_mask.any(dim=1)
        first_true_indices[no_true_mask] = non_prompt_mask.size(1)
        batch_size, seq_len = non_prompt_mask.size()
        arange = torch.arange(seq_len, device=non_prompt_mask.device).unsqueeze(0).expand(batch_size, -1)
        prompt_mask = arange < first_true_indices.unsqueeze(1)
        llm_inputs.batch["prompt_mask"] = prompt_mask
        llm_inputs.batch["scores"] = score_tensor
        # for llm raw response
        llm_raw_text_list, _ = self._format_messages(
            env_output=self.rollout_cache, prepare_for_update=True, use_raw_llm_response=True
        )
        llm_inputs.non_tensor_batch["turn_scores"] = np.array(scores, dtype=object)
        llm_inputs.non_tensor_batch["episode_scores"] = np.array(episode_scores, dtype=object)
        llm_inputs.non_tensor_batch["llm_raw_text_list"] = np.array(llm_raw_text_list, dtype=object)

        entry = self.env_entry
        status = entry["status"]
        env_metric = {
            "success": float(status.terminated and (not status.truncated)),
            "num_actions": status.num_actions,
        }
        custom_metric = {}

        known_string_info_keys = [
            "step_execution_error",
            "tool_error_details",
            "step_critical_error",
            "step_runtime_error",
            "message", # Common key from LocalToolExecutor.latest_status_info
            "error"    # Common key for errors (e.g. from DeepResearchEnv.step initial error return)
            # Add other known string keys from info dicts if necessary
        ]

        for turn in self.rollout_cache["history"]:
            for k, v in turn.get("info", {}).items():
                if k == "success" or k in known_string_info_keys: # Skip 'success' and known string fields
                    continue

                if k not in custom_metric:
                    custom_metric[k] = []

                if isinstance(v, (int, float)):
                    custom_metric[k].append(float(v))
                elif isinstance(v, str):
                    try:
                        metric_val = float(v)
                        custom_metric[k].append(metric_val)
                    except ValueError:
                        self.logger.debug(
                            f"Info field '{k}' with string value '{v}' could not be converted to float. Skipping for custom_metric."
                        )
                # Other data types (like bool, list, dict) are implicitly skipped for custom_metric float aggregation

        for k, v in custom_metric.items():
            env_metric[k] = np.sum(v) / len(self.rollout_cache["history"])

        self.rollout_cache["history"][-1]["metrics"] = custom_metric
        env_metric = {f"env/{entry['tag']}/{k}": v for k, v in env_metric.items()}
        env_metric["env/response_length"] = response_length
        self.rollout_cache["metrics"] = env_metric
        llm_inputs.meta_info = {"metrics": env_metric}
        return llm_inputs

    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment"""
        if isinstance(state, str):  # text state
            return state
        elif isinstance(
            state, np.ndarray
        ):  # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode="RGB") for _state in state]
        return results

    def _update_cache_history(
        self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None
    ):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None:  # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)

        entry = {}  # append state to history
        if isinstance(next_state, str):  # text state
            entry["state"] = next_state
        else:  # multimodal state
            entry["state"] = "<images>" * len(next_state)
            entry["images"] = next_state
        entry["actions_left"] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry["env"].config, "action_lookup", None)
        if action_lookup is None:
            mapped_actions = actions
        else:  # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions

    def _execute_actions(self, env, actions):
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        for action in actions:
            _, reward, done, info = env.step(action)
            acc_reward += reward
            turn_info.update(info)  # NOTE: currently use last info for multi-action
            executed_actions.append(action)
            if done:
                turn_done = True
                break
        return acc_reward, turn_info, turn_done, executed_actions

    def _log_env_state(
        self,
        status,
        history,
        cur_obs,
        max_actions_per_traj,
        executed_actions,
        all_actions,
        acc_reward,
        turn_done,
        turn_info,
        env_input,
    ) -> Tuple[EnvStatus, List[Dict]]:
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward)
        actions_left = max_actions_per_traj - status.num_actions
        if turn_done:
            status.terminated = True
            status.truncated = not turn_info.get("success", False)
        history = self._update_cache_history(
            history,
            next_state=obs,
            actions_left=actions_left,
            num_actions_info={
                "actions": executed_actions,
                "reward": acc_reward,
                "info": turn_info,
                "llm_response": env_input["llm_response"],
                "llm_raw_response": env_input["llm_raw_response"],
            },
        )
        return status, history

    def _format_messages(self, env_output: Dict, prepare_for_update: bool, use_raw_llm_response: bool):
        if env_output["history"] and "state" in env_output["history"][-1] and (not use_raw_llm_response and prepare_for_update):
            env_output["history"] = env_output["history"][
                :-1
            ]  # when prepare for update, we do not add the state from the n+1 turn to the trajectory

        entry = self.env_entry # Get the current environment entry
        is_deepresearch_env = (entry.get("tag") == "DeepResearchEnv")

        if is_deepresearch_env:
            system_prompt_content = None
            if hasattr(self.env_entry["env"], 'get_system_prompt'):
                system_prompt_content = self.env_entry["env"].get_system_prompt()

            if not system_prompt_content: # Check if None or empty
                self.logger.warning(f"Failed to get live system prompt for DeepResearchEnv (env_id: {env_output.get('env_id', 'N/A')}). Attempting fallback.")
                # Check attribute existence and content before assigning
                if hasattr(self.env_entry["status"], 'initial_system_prompt') and self.env_entry["status"].initial_system_prompt:
                    system_prompt_content = self.env_entry["status"].initial_system_prompt

                if not system_prompt_content: # Check again if None or empty
                    self.logger.warning(f"Fallback system prompt also unavailable for DeepResearchEnv (env_id: {env_output.get('env_id', 'N/A')}). Using hardcoded default.")
                    system_prompt_content = "You are a helpful AI assistant. Your primary goal is to complete the given task using available tools."

            task_desc_content = None
            # Check attribute existence and content before assigning
            if hasattr(self.env_entry["status"], 'task_description') and self.env_entry["status"].task_description:
                task_desc_content = self.env_entry["status"].task_description

            if not task_desc_content: # Fallback for task_description if it somehow ended up empty
                self.logger.warning(f"Task description is empty for DeepResearchEnv (env_id: {env_output.get('env_id', 'N/A')}). Using default.")
                task_desc_content = "Please complete the assigned research task."

            messages = [
                {"role": "system", "content": system_prompt_content},
                # The task_description is the first user message.
                # It's already in history as the first 'state' if using the old logic for next_state in reset.
                # With the new reset logic, task_description is passed to env, and also set in status.
                # The history will start with the actual task description as the first 'state'.
                # So, the loop below will pick it up.
                # No, the first history item for DeepResearchEnv is the task_description (as 'state').
                # The _format_messages logic should construct the initial user message from task_desc_content
                # and then process the history, skipping this first state if it's the task description.
                # Let's ensure the task description is the first user message.
            ]
            if task_desc_content: # Ensure task_description is present
                 messages.append({"role": "user", "content": task_desc_content})

            current_history = env_output["history"]
            # For DeepResearchEnv, the first item in history is the task_description (logged as 'state').
            # We've already added it as the first user message if we use the logic above.
            # Or, if we assume history starts *after* the initial task_description:
            # The history items are:
            # 1. (Optional, if logged this way) Initial state: task_description
            # 2. LLM Response (tool call), Reward
            # 3. Env State (<tool_response>), Reward
            # ...

            # Revised loop for DeepResearchEnv history:
            # The first 'state' in history for DeepResearchEnv is the task_description.
            # We've already used entry["status"].task_description for the first user message.
            # So, the loop should start from the first actual turn (assistant's tool call).

            for content_item in current_history:
                # If the first history item is the task description, we might want to skip it here
                # if task_desc_content was already added.
                # Let's assume current_history here means actual interaction history *after* initial setup.
                # The _update_cache_history adds the task_description as the first 'state'.
                # So, the first content_item WILL have 'state' as task_description.

                if "llm_response" in content_item: # Assistant's turn (tool call)
                    messages.append({
                        "role": "assistant",
                        "content": content_item["llm_response"] # This should be the <tool_call>...</tool_call>
                    })

                # The 'state' field from history is the observation from the env after assistant's action.
                if "state" in content_item and content_item["state"]:
                    # For DeepResearchEnv, the first state is the task_description.
                    # We need to ensure we don't add it again if already added,
                    # or handle it if it's a tool_response.
                    if content_item["state"] == task_desc_content and messages[-1]["role"] == "user" and messages[-1]["content"] == task_desc_content:
                        # This is the initial task description, already added as the first user message. Skip.
                        pass
                    elif content_item["state"].strip().startswith("<tool_response>"):
                         messages.append({
                             "role": "tool",
                             "content": content_item["state"].strip()
                         })
                    # else:
                        # Non-tool-response state from DeepResearchEnv. Could be a final message or error.
                        # For now, these are not explicitly formatted into the LLM prompt here,
                        # as the primary interaction loop is tool_call -> tool_response.
                        # If DeepResearchEnv produces other kinds of states that need to go to LLM,
                        # this part would need specific handling.
                        # self.logger.warning(f"DeepResearchEnv: Unhandled state in history: {content_item['state'][:100]}")

                if "reward" in content_item and not (prepare_for_update and content_item is env_output["history"][-1]):
                     messages.append({"role": "user", "content": f"Reward:\n{content_item['reward']}\n"})
        else:
            # Existing logic for other environments:
            messages = [
                {
                    "role": "system",
                    "content": "You're a helpful assistant. You are a good game player. You are aiming to get high reward in the game.",
                },
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]},
            ]

            for idx, content in enumerate(env_output["history"]):
                messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                if "state" in content:
                    FORMAT_PROMPT = (
                        "<think> [Your thoughts] </think> <answer> [your answer] </answer>"
                        if self.pipeline_config.enable_think
                        else "<answer> [your answer] </answer>"
                    )
                    LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                    messages[-1]["content"] += (
                        f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. "
                        f"Always output: {FORMAT_PROMPT} with no extra text."
                        f"Strictly follow this format, history response that do not follow the format will be set as 'INVALID'. {LENGTH_PROMPT}\n"
                        f"Decide the next action:\n"
                    )
                if "llm_raw_response" in content:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": (
                                content["llm_response"] if not use_raw_llm_response else content["llm_raw_response"]
                            ),
                        }
                    )
                if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                    messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})

            # NOTE: this assertion is important for loss mask computation for non-DeepResearch envs
            if messages[1]["role"] == "user": # Standard envs start with user prompt after system
                 assert all(msg["role"] == "assistant" for msg in messages[2::2])


        if use_raw_llm_response: # This part seems general, may need review for DeepResearchEnv
            # For DeepResearchEnv, if messages start [system, user(task_desc), assistant(tool_call), tool(tool_resp)]
            # messages[2:] would be [assistant, tool, ...]. This might be okay if "raw" means "for PPO value func"
            # where only assistant responses are primary.
            # However, the original assertion `all(msg["role"] == "assistant" for msg in messages[2::2])`
            # will fail for DeepResearchEnv due to "tool" role.
            # This needs careful consideration of what `use_raw_llm_response=True` implies.
            # If it's for creating a prompt for a value function that only sees assistant turns,
            # then we'd need to filter out "tool" roles for that specific case.
            # For now, let's assume this part is mostly for non-DeepResearch or that the assertion will be conditional.
            if entry["tag"] != "DeepResearchEnv":
                 assert all(msg["role"] == "assistant" for msg in messages[2::2]) # Keep assertion for non-DeepResearch
            messages = messages[2:] # This will also fail for DeepResearchEnv if it expects system, user.
                                    # This line is likely intended for non-DeepResearch envs if it means "get only assistant turns".
                                    # Let's make this conditional too.
            if entry["tag"] != "DeepResearchEnv":
                messages = messages[2:] # Original logic for non-DeepResearch
            else:
                # For DeepResearchEnv, if use_raw_llm_response is true, what should be returned?
                # If it's for a value model, it might want system, user, assistant, tool, assistant, tool ...
                # or it might want a simplified history. This is unclear from current context.
                # Let's assume for now `use_raw_llm_response` is NOT typically used with DeepResearchEnv
                # or that its handling here needs more specific requirements for DeepResearchEnv.
                # The safest thing is to let it pass through and see if downstream components complain.
                pass # No slicing for DeepResearchEnv for now, return all formatted messages.


        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=(not prepare_for_update), tokenize=False
        )
        if not prepare_for_update:
            if self.pipeline_config.enable_think:
                text += "<think>"  # force the LLM to think before answering
            else:
                text += "<answer>"  # force the LLM to answer

        # TODO: 应该没有必要，注意处理mask
        text = text.replace("<|im_end|>\n", "<|im_end|>")
        return [text], [messages]

    def _init_prefix_lookup(self):
        # This method populates:
        # 1. self.deepresearch_tasks_by_split: For DeepResearchEnv, loads tasks into "train" and "validation" lists.
        # 2. self.prefix_lookup: Maps each global env_id to its initial instruction string (for non-DeepResearch or fallback).
        # 3. self.env_config_lookup: Maps each global env_id to its specific config details (e.g., max_tokens).

        # --- Part 1: Prepare per-tag information from global custom_envs ---
        # `tag_specific_prefixes` will store the base instruction string for each env_tag.
        # `tag_specific_env_configs` will store other specific configurations like max_tokens for each env_tag.
        tag_specific_prefixes = {}
        tag_specific_env_configs = {}

        # Helper function to extract instructions from a list of task items
        def extract_instructions(task_items: List[Any], logger_ref, env_tag_for_log: str) -> List[str]: # Added logger_ref and env_tag_for_log for context
            extracted: List[str] = []
            if isinstance(task_items, list):
                for item_idx, item in enumerate(task_items):
                    if isinstance(item, dict):
                        instruction = item.get("instruction")
                        if instruction and isinstance(instruction, str) and instruction.strip():
                            extracted.append(instruction.strip())
                        # else:
                        #    logger_ref.debug(f"Task item {item_idx} for env_tag '{env_tag_for_log}' is not a dict with a valid 'instruction' string.")
                    # else:
                    #    logger_ref.debug(f"Task item {item_idx} for env_tag '{env_tag_for_log}' is not a dictionary.")
            # else:
            #    if task_items: # Only log if it's not an intentional empty list
            #        logger_ref.debug(f"Task items for env_tag '{env_tag_for_log}' is not a list, but {type(task_items)}.")
            return extracted

        # Initialize self.deepresearch_tasks_by_split for all relevant tags first.
        for tag_name, config_from_yaml in self.pipeline_config.custom_envs.items():
            temp_default_config = REGISTERED_ENV_CONFIGS[config_from_yaml.env_type]()
            temp_merged_config = asdict(temp_default_config)
            temp_merged_config.update(config_from_yaml)

            if temp_merged_config.get("env_type") == "deepresearch":
                if tag_name not in self.deepresearch_tasks_by_split:
                    self.deepresearch_tasks_by_split[tag_name] = {"train": [], "validation": []}

        for env_tag, env_config_from_yaml in self.pipeline_config.custom_envs.items():
            default_env_type_config = REGISTERED_ENV_CONFIGS[env_config_from_yaml.env_type]()
            env_config_new = asdict(default_env_type_config)
            env_config_new.update(env_config_from_yaml)

            base_env_instruction = env_config_new.get("env_instruction", "")

            if env_config_new.get("env_type") == "deepresearch":
                actual_env_params = env_config_new.get("env_config", {})
                task_desc_file_path = actual_env_params.get("task_description_file")

                train_tasks: List[str] = []
                validation_tasks: List[str] = []

                if task_desc_file_path and isinstance(task_desc_file_path, str):
                    if ".." in task_desc_file_path:
                        self.logger.warning(
                            f"Task description file path for '{env_tag}' ('{task_desc_file_path}') contains '..'. Skipping file load."
                        )
                    else:
                        self.logger.info(f"Attempting to load tasks for DeepResearchEnv '{env_tag}' from file: {task_desc_file_path}")
                        try:
                            with open(task_desc_file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            if isinstance(data, dict):
                                train_tasks = extract_instructions(data.get("train", []), self.logger, env_tag)
                                validation_tasks = extract_instructions(data.get("validation", []), self.logger, env_tag)
                                if train_tasks or validation_tasks:
                                    self.logger.info(f"Loaded {len(train_tasks)} train tasks and {len(validation_tasks)} validation tasks for '{env_tag}'.")
                                else:
                                    self.logger.warning(f"Task file '{task_desc_file_path}' for '{env_tag}' (dict format) had no valid tasks under 'train' or 'validation'.")
                            elif isinstance(data, list):
                                self.logger.info(f"Task file for '{env_tag}' is in legacy list format. Treating all tasks as training tasks.")
                                train_tasks = extract_instructions(data, self.logger, env_tag)
                                if train_tasks:
                                    self.logger.info(f"Loaded {len(train_tasks)} tasks as training tasks (legacy format) for '{env_tag}'.")
                                else:
                                    self.logger.warning(f"Task file '{task_desc_file_path}' for '{env_tag}' (list format) yielded no valid tasks.")
                            else:
                                self.logger.warning(f"Invalid JSON structure in '{task_desc_file_path}' for '{env_tag}'. Expected dict or list, got {type(data)}.")
                        except FileNotFoundError:
                            self.logger.warning(f"Task description file not found for '{env_tag}': {task_desc_file_path}.")
                        except json.JSONDecodeError:
                            self.logger.warning(f"Invalid JSON in task description file '{task_desc_file_path}' for '{env_tag}'.")
                        except Exception as e:
                            self.logger.error(f"Error reading/parsing task file '{task_desc_file_path}' for '{env_tag}': {e}")

                # Ensure the entry for the tag exists from pre-initialization
                self.deepresearch_tasks_by_split[env_tag]["train"] = train_tasks
                self.deepresearch_tasks_by_split[env_tag]["validation"] = validation_tasks

                if not train_tasks and not validation_tasks:
                     self.logger.warning(f"No tasks loaded from file for DeepResearchEnv '{env_tag}'. Fallback instruction may be used.")

                if not base_env_instruction:
                    base_env_instruction = "Default DeepResearch task. Task file might be missing or empty."
                tag_specific_prefixes[env_tag] = base_env_instruction

            else:
                if env_config_new.get("grid_vocab", False):
                    grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join(
                        [f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()]
                    )
                    base_env_instruction += grid_vocab_str
                if env_config_new.get("action_lookup", False):
                    action_lookup_str = "\nYour available actions are:\n" + ", ".join(
                        [f"{v}" for k, v in env_config_new["action_lookup"].items()]
                    )
                    base_env_instruction += action_lookup_str
                tag_specific_prefixes[env_tag] = base_env_instruction

            tag_specific_env_configs[env_tag] = {
                "max_tokens": env_config_new.get("max_tokens", self.pipeline_config.response_length)
            }

        temp_prefix_lookup = {}
        temp_env_config_lookup = {}
        global_env_id_counter = 0

        managers_to_process = []
        if hasattr(self.pipeline_config, 'train_env_manager') and self.pipeline_config.train_env_manager:
            managers_to_process.append(self.pipeline_config.train_env_manager)
        if hasattr(self.pipeline_config, 'val_env_manager') and self.pipeline_config.val_env_manager:
            managers_to_process.append(self.pipeline_config.val_env_manager)

        for manager_cfg in managers_to_process:
            for tag_idx, env_tag_for_mgr in enumerate(manager_cfg.tags):
                num_groups_for_tag = manager_cfg.n_groups[tag_idx]
                num_envs_for_tag_in_mgr = num_groups_for_tag * manager_cfg.group_size

                for _ in range(num_envs_for_tag_in_mgr):
                    if env_tag_for_mgr in tag_specific_prefixes:
                        temp_prefix_lookup[global_env_id_counter] = tag_specific_prefixes[env_tag_for_mgr]

                    if env_tag_for_mgr in tag_specific_env_configs:
                        temp_env_config_lookup[global_env_id_counter] = tag_specific_env_configs[env_tag_for_mgr]
                    elif env_tag_for_mgr in tag_specific_prefixes: # Fallback if not in specific env_configs (e.g. if only prefix was set)
                         self.logger.warning(f"Env tag '{env_tag_for_mgr}' for env_id {global_env_id_counter} not found in tag_specific_env_configs, check max_tokens setup.")
                         temp_env_config_lookup[global_env_id_counter] = {"max_tokens": self.pipeline_config.response_length} # Basic fallback
                    else:
                        self.logger.error(f"Env tag '{env_tag_for_mgr}' used in a manager but not found in custom_envs processing for env_id {global_env_id_counter}.")
                        # Provide a very basic fallback to prevent crashes, though this indicates a config issue.
                        temp_prefix_lookup[global_env_id_counter] = "Configuration error: Unknown environment tag."
                        temp_env_config_lookup[global_env_id_counter] = {"max_tokens": self.pipeline_config.response_length}

                    global_env_id_counter += 1

        self.prefix_lookup = temp_prefix_lookup
        self.env_config_lookup = temp_env_config_lookup

    def _parse_response(self, response: str) -> Tuple[str, List[str]]:
        actions: List[str] = []
        llm_response_for_history: str = response # Default: raw response for history

        # Determine if the current environment is DeepResearchEnv
        is_deepresearch_env = (hasattr(self, 'env_entry') and self.env_entry and self.env_entry.get("tag") == "DeepResearchEnv")

        if is_deepresearch_env:
            # DeepResearchEnv: Expects <tool_call> tags for actions, multiple calls possible.
            # Narrative is anything outside these tags. llm_response_for_history remains raw.
            tool_call_contents = re.findall(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            if tool_call_contents:
                cleaned_tool_calls = [tc.strip() for tc in tool_call_contents if tc.strip()] # Ensure content is not just whitespace
                if cleaned_tool_calls: # Ensure list is not empty after stripping and filtering
                    actions = [json.dumps(cleaned_tool_calls)] # Serialize list of tool calls
            # If no tool_call_contents or if all were empty strings, actions remains []
            # llm_response_for_history is already set to the raw response
        else:
            # Logic for other environments (prioritize single <tool_call>, then <answer>)
            tool_call_match = re.search(r"<tool_call>(.*?)</tool_call>", response, re.DOTALL)
            if tool_call_match:
                action_content = tool_call_match.group(1).strip()
                if action_content: # Ensure extracted content is not empty
                    actions = [action_content]
                # llm_response_for_history is already set to the raw response
            else:
                # Try to parse <answer> tags if no <tool_call> was found
                think_pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"
                answer_pattern = r"<answer>(.*?)</answer>"

                match = None
                think_content_for_history = ""

                if self.pipeline_config.enable_think:
                    match = re.search(think_pattern, response, re.DOTALL)
                    if match:
                        think_content_for_history = match.group(1).strip() # Assign only if enable_think and think_pattern matched

                if not match:
                    match = re.search(answer_pattern, response, re.DOTALL)

                if match:
                    action_content_from_match = ""
                    if self.pipeline_config.enable_think and len(match.groups()) == 2 and think_content_for_history: # Matched think_pattern and think_content was captured
                        action_content_from_match = match.group(2).strip()
                    elif not self.pipeline_config.enable_think and len(match.groups()) == 1: # Matched answer_pattern (and think disabled)
                        action_content_from_match = match.group(1).strip()
                    elif len(match.groups()) == 1 and not think_content_for_history : # Fallback if only answer part of think_pattern matched or answer_pattern itself
                         action_content_from_match = match.group(1).strip()


                    # Strip special tokens from captured contents for history reconstruction
                    temp_action_content = action_content_from_match
                    for special_token in self.pipeline_config.special_token_list:
                        temp_action_content = temp_action_content.replace(special_token, "").strip()
                        if self.pipeline_config.enable_think and think_content_for_history:
                            think_content_for_history = think_content_for_history.replace(special_token, "").strip()

                    parsed_actions_list = [act.strip() for act in temp_action_content.split(self.pipeline_config.action_sep) if act.strip()]

                    max_actions = 1

                    processed_action_content_for_history = ""
                    if parsed_actions_list:
                        actions = parsed_actions_list[:max_actions]
                        processed_action_content_for_history = (" " + self.pipeline_config.action_sep + " ").join(actions)
                    else:
                        actions = []
                        # processed_action_content_for_history remains "" if answer tag was empty or yielded no actions

                    # Reconstruct llm_response_for_history for <answer> cases
                    if self.pipeline_config.enable_think: # Use the potentially stripped think_content_for_history
                        llm_response_for_history = f"<think>{think_content_for_history}</think><answer>{processed_action_content_for_history}</answer>"
                    else:
                        llm_response_for_history = f"<answer>{processed_action_content_for_history}</answer>"
                # else: No <tool_call> and no <answer> found for non-DeepResearch env.
                # llm_response_for_history remains raw response, actions remains []. This is correct.

        return llm_response_for_history, actions

    def start_input_queue_process(self):
        def process_input_queue(input_queue):
            while True:
                command = input_queue.get()
                if command == "stop":
                    self.logger.info(f"{self.worker_name} stopped, episode_id: {self.episode_id}")
                    self.running = False
                    ray.get(self.generate_scheduler.abort_request.remote(DataProto()))
                    break

        self.process_input_queue_thread = Thread(target=process_input_queue, args=(self.input_queue,))
        self.process_input_queue_thread.start()
