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
import torch
from transformers import PreTrainedTokenizerFast, AutoTokenizer

from roll.datasets.collator import DataCollatorWithPaddingForPaddedKeys


def test_data_collator_with_padding_for_padded_keys():
    tokenizer = AutoTokenizer.from_pretrained("/Users/pan/Downloads/huggingface/gpt2-imdb", padding_side="left")

    tokenizer.pad_token_id = tokenizer.eos_token_id

    max_length = 32
    data_collator = DataCollatorWithPaddingForPaddedKeys(
        tokenizer=tokenizer, padding="max_length", max_length=max_length
    )

    features = [
        {
            "input_ids": tokenizer.encode("Hello, how are you?", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(1),
            "auxiliary": {"type": 1},
        },
        {
            "input_ids": tokenizer.encode("I'm fine, thank you!", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(0),
            "auxiliary": {"type": 2},
        },
        {
            "input_ids": tokenizer.encode("What about you?", return_tensors="pt").squeeze(0),
            "labels": torch.tensor(1),
            "auxiliary": {"type": 3},
        },
    ]
    for feature in features:
        feature["attention_mask"] = [1] * len(feature["input_ids"])

    batch = data_collator(features)

    print("Padded input_ids:")
    print(batch["input_ids"])
    print("Padded attention_mask:")
    print(batch["attention_mask"])
    print("Labels:")
    print(batch["labels"])

    assert (
        batch["input_ids"].shape[1] == max_length
    ), f"Expected max_length {max_length}, got {batch['input_ids'].shape[1]}"
    print(f"All inputs padded to length {max_length} correctly.")
