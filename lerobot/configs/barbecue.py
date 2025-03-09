#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import datetime as dt
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from lerobot.common import (
    policies,  # noqa: F401
)
from lerobot.configs import parser
from lerobot.common.datasets.transforms import ImageTransformsConfig
from torchvision.transforms import v2
from lerobot.configs.policies import PreTrainedConfig

transform = v2.Compose([
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                v2.RandomPerspective(distortion_scale=0.5),
                v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
                v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
                v2.Resize((16 * 14, 22 * 14)),
                # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                # v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])



@dataclass
class DatasetConfig:
    # You may provide a list of datasets here. `train.py` creates them all and concatenates them. Note: only data
    # keys common between the datasets are kept. Each dataset gets and additional transform that inserts the
    # "dataset_index" into the returned item. The index mapping is made according to the order in which the
    # datasets are provided.
    repo_id: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | None = None
    episodes: list[int] | None = None
    image_transforms = transform
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = "pyav"


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self):
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class EvalPipelineConfig:
    # Either the repo ID of a model hosted on the Hub or a path to a directory containing weights
    # saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch
    # (useful for debugging). This argument is mutually exclusive with `--config`.
    policies: List[PreTrainedConfig] | None = None
    output_dir: Path | None = None
    job_name: str | None = None
    seed: int | None = 1000

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        for policy in self.policies:
            policy_path = parser.get_path_arg("policy")
            if policy_path:
                policy = PreTrainedConfig.from_pretrained(policy_path)
                policy.pretrained_path = policy_path

            else:
                logging.warning(
                    "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
                )

        if not self.output_dir:
            now = dt.datetime.now()
            eval_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/eval") / eval_dir

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
