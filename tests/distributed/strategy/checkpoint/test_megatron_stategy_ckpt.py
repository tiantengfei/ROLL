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
from typing import Any

import torch

from roll.pipeline.base_worker import ActorWorker
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.initialize import init
from roll.models.model_providers import default_tokenizer_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.utils.logging import get_logger
from tests.distributed.strategy.make_baseline_config import make_baseline_config

logger = get_logger()


class TestModelCheckpointPipeline(BasePipeline):

    def __init__(self, pipeline_config: RLVRConfig):
        super().__init__(pipeline_config)

        self.tokenizer = default_tokenizer_provider(
            model_args=self.pipeline_config.actor_train.model_args,
            template_name=self.pipeline_config.actor_train.data_args.template,
        )
        max_steps = 10240 * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=ActorWorker,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=True)
        self.set_checkpoint_clusters(self.actor_train)

    @torch.no_grad()
    def run(self):
        # self.actor_train.strategy.save_checkpoint(self.pipeline_config.output_dir, global_step)
        self.do_checkpoint(global_step=1)
        self.do_checkpoint(global_step=2)


if __name__ == "__main__":
    init()

    ppo_config = make_baseline_config(config_path="./checkpoint", config_name="megatron_config")

    pipeline = TestModelCheckpointPipeline(ppo_config)
    metric_list = pipeline.run()
