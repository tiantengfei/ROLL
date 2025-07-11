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
import gc
from typing import Optional

import torch
from vllm.worker.worker import Worker
from vllm.device_allocator.cumem import CuMemAllocator

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger

logger = get_logger()


class Worker073(Worker, WorkerHelper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_loaded : bool = True
        self.kv_cache_loaded : bool = True

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

    def reload_model(self):
        if not self.weight_loaded:
            self.wake_up(["weights"])
            self.weight_loaded = True

    def load_states(self):
        self.reload_model()
        if not self.kv_cache_loaded:
            self.wake_up(["kv_cache"])
            self.kv_cache_loaded = True

    def offload_states(self):
        assert (self.weight_loaded and self.kv_cache_loaded) or (not self.weight_loaded and not self.kv_cache_loaded)
        if not self.weight_loaded:
            return
        self.sleep()
        self.weight_loaded = False
        self.kv_cache_loaded = False
        if hasattr(self, 'recv_manager'):
            self.recv_manager.clear()
        gc.collect()
        torch.cuda.empty_cache()
