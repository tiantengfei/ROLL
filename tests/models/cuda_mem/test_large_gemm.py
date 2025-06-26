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
import torch.nn as nn

# Define parameters
vocab_size = 128 * 1000  # Vocabulary size
intermediate_size = 1560  # Intermediate hidden layer size
batch_size = 1
seq_len = 4096  # Assumed sequence length

# Create a random hidden layer output tensor (batch_size, seq_len, intermediate_size)
hidden_output = torch.randn(batch_size, seq_len, intermediate_size).cuda()

# Create the last linear layer, mapping hidden layer size to vocabulary size
# linear_layer = nn.Linear(intermediate_size, vocab_size).cuda()

# Compute the last layer for each time step, can use reshape
# If computing by seq_len, can use batched as (batch_size * seq_len, intermediate_size)
# logits = linear_layer(hidden_output.view(-1, intermediate_size))  # Reshape to (batch_size * seq_len, intermediate_size)

# Directly construct weight matrix W (intermediate_size, vocab_size)
weight_matrix = torch.randn(intermediate_size, vocab_size).cuda()
# Compute logits, perform matrix multiplication
# Compute the last layer for each time step, can use reshape
logits = torch.matmul(hidden_output.view(-1, intermediate_size), weight_matrix)


# Reshape logits to (batch_size, seq_len, vocab_size)
logits = logits.view(batch_size, seq_len, vocab_size)

# Compute softmax to get probability distribution
probabilities = nn.functional.softmax(logits, dim=-1)  # Compute probability distribution for each time step

del logits, probabilities, weight_matrix, hidden_output
torch.cuda.empty_cache()
