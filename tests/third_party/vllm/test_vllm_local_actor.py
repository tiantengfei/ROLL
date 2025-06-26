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
import json
import os
import pickle

import ray
from ray.runtime_env import RuntimeEnv
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm import SamplingParams

from roll.distributed.scheduler.resource_manager import ResourceManager
from roll.third_party.vllm import LLM


model_path = "Qwen/Qwen2.5-7B-Instruct"

prompts = [
    "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞,生成一段文案",
    "根据关键词描述生成女装/女士精品行业连衣裙品类的发在淘宝的小红书风格的推送配文，包括标题和内容。关键词：pe。要求:1. 推送标题要体现关键词和品类特点，语言通顺，有吸引力，约10个字；2. 推送内容要语言通顺，突出关键词和品类特点，对目标受众有吸引力，长度约30字。标题:",
    "100.25和90.75谁更大？",
]


def chat_format(prompt):
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

chat_prompts = []
for prompt in prompts:
    chat_prompts.append(chat_format(prompt))

# os.environ["RAY_DEBUG"] = "legacy"

# breakpoint()
runtime_env = {
    "env_vars": {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCHINDUCTOR_COMPILE_THREADS": "2",
        # "RAY_DEBUG": "legacy",
        "NCCL_CUMEM_ENABLE": "0",  # https://github.com/NVIDIA/nccl/issues/1234
        "NCCL_NVLS_ENABLE": "0",
    }
}
ray.init(log_to_driver=True, runtime_env=runtime_env)
resource_manager = ResourceManager()
placement_groups = resource_manager.allocate_placement_group(world_size=1, device_mapping=[0])


@ray.remote
class TestActor:
    def __init__(self, placement_groups):
        self.model = LLM(
            resource_placement_groups=placement_groups[0],
            model=model_path,
            block_size=16,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
            tensor_parallel_size=1,
            trust_remote_code=True,
            distributed_executor_backend="ray",
            disable_custom_all_reduce=True,
            enforce_eager=True,
            enable_sleep_mode=True,
        )

    def run(self):
        sampling_params = SamplingParams(temperature=0.0, top_p=0.99, top_k=100, max_tokens=512)
        self.model.offload_states()
        import torch

        print(f"memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3}")

        # use torch.cuda.mem_get_info()[0] in sleep mode: https://github.com/vllm-project/vllm/pull/11743
        print(f"free: {torch.cuda.mem_get_info()[0] / 1024 ** 3}")
        import pdb

        pdb.set_trace()

        self.model.load_states()

        vllm_outputs = self.model.generate(
            sampling_params=sampling_params,
            prompts=chat_prompts,
        )

        print(vllm_outputs)


env_vars = {
    "WORLD_SIZE": str(1),
    "RANK": str(0),
    "LOCAL_RANK": str(0),
    "CLUSTER_NAME": "",
    "WORKER_NAME": "",
}
env_vars.update(
    {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, list(range(0, 8)))),
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    }
)
runtime_env = RuntimeEnv(env_vars=env_vars)

actor = TestActor.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=placement_groups[0][0]["placement_group"]),
    name="actor",
    runtime_env=runtime_env,
    num_cpus=0.01,
    num_gpus=0.01,
).remote(placement_groups=placement_groups)
ray.get(actor.run.remote())

ray.shutdown()
