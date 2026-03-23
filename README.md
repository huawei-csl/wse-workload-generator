
# WSE Workload Generator

A flexible workload generator for simulating and analyzing parallelism strategies in large-scale AI models on wafer-scale engines.

## Features

- **Supported Parallelism Strategies:**
  - Data Parallel (DP)
  - Tensor Parallel (TP)
  - Pipeline Parallel (PP)
  - Sequence Parallel (SP)
  - Expert Parallel (EP)

- **Layer Modeling:**
  - Linear
  - Self-attention (GQA, MLA-naive, MLA-absorb)
  - Allgather
  - Allreduce
  - Alltoall
  - Multicast

- **Architectures:**
  - DeepSeekv3
  - Llama GQA (in progress)

- **MoE Workload Model:**
    - Identical: Each expert receives identical number of tokens
    - Uniform: The number of tokens routed to each expert is sampled from a uniform distribution
    - Empirical MMLU: The number of tokens routed to each expert is sampled from data collected during the inference of MMLU dataset

## Folder structure
```
wse-workloads/
├── configs/                # Model and system configuration files (e.g., config.json, system.json)
├── output/
│   ├── graph/              # Compute graph outputs
│   ├── nodes/              # Node-level workload CSV outputs
│   ├── traces/             # Core-level trace outputs
│   └── visuals/            # Visualization PNG files
├── scripts/                # Example and helper scripts (e.g., deepseekv3.sh)
├── src/
│   ├── generate_nodes.py   # Node-level workload partitioning script
│   ├── generate_traces.py  # Core-level trace generation script
│   ├── visualize_traces.py # Core-level visualization script
│   ├── node_level/         # Node-level source code
│   ├── core_level/         # Core-level source code
│   └── visualize/          # Visualization related source code
├── README.md               # Project documentation
└── setup.py                # Python install script
```

## Installation
In an environment with ```python>=3.7```, run:
```bash
pip install -e .
```

## Quick start
Run the following script to generate traces for specified layers of Deepseekv3:
```bash
bash scripts/deepseekv3.sh
```

## How to use
1) Make sure the **config.json** (downloaded from HF) for the target model is stored under configs/
2) Enter system properties in **system.json**

### Node-level workload partitioning
Run src/generate_nodes.py with the desired arguments. For example:   
```
python src/generate_nodes.py --model_config configs/deepseekv3.json --system_config configs/system.json --bsz 1024 --prefill_len 2048 --decode_len 10 --only_decode 1 --simplified_decode 1 --dtype fp16
```

The tool will generate various system requirements (i.e., **# of flops, memory reads, network requirements**) for each node and for each inference step (prefill + decode) as csv files under **output/nodes**.

Additionally, it will create a compute graph and save it under **output/graph**.

For argument descriptions, run:
```
python src/generate_nodes.py --help
```

### Core-level trace generation
Run the core-level trace generator:
```
python src/generate_traces.py --layers decode5 --iter decode0 --dtype fp16
```

The trace generator will generate core-level traces under **output/traces**

### Data movement pattern visualization
Run the visualization tool:
```
python src/visualize_traces.py --layers decode5 --iter decode0
```

The visualization tool will generate a png file for each layer under **output/visuals**.

## Unit tests
Run unit tests:
```
py.test
```

## MoE Communication Types

### Unicast

Point-to-point communication between two nodes. 

The generated CSV files has a row for each unicast operation in the following format:

- Network data (B): Size of the tensor to be sent in bytes
- Comm. group: Destination node id
- Dimensions: Tensor dimensions

For example, if node 3 sends a tensor of [4,1,32] in fp16 to node 5, the following row will be added to node3/decode*.csv:

*_unicast;Unicast;0;0;0;256;5;[4, 1, 32]

NOTE: if src and dst can be the same. In this case, network data will be equal to zero. 

### Multicast

Sending a copy of a tensor from one source to multiple.

The generated CSV files has a row for each multicast operation in the following format:

- Network data (B): Size of the tensor to be multicast in bytes
- Comm. group: A list of destination node ids
- Dimensions: Tensor dimensions

For example, if node 3 sends a tensor of [4,1,32] in fp16 to nodes 5, 6, and 8, the following row will be added to node3/decode*.csv:

*_multicast;Multicast;0;0;0;256;[5, 6, 8];[4, 1, 32]

### Allreduce

A group of nodes sum-reduce their intermediate results. Assume node $i$ has tensor $T_i$, at the end of the allreduce operations, each node has $\sum_{j=1}^N{T_j}$, where N is the number of nodes.

The generated CSV files has a row for each allreduce operation in the following format:

- Network data (B): Size of the tensor to be sum-reduced in bytes
- Comm. group: A list of node ids that participate in allreduce
- Dimensions: Tensor dimensions

For example, if node 4, 5, 6, 7 perform sum-reduce for a tensor of [4,1,32] in fp16, the following row will be added to node4-7/decode*.csv:

*_allreduce;Allreduce;0;0;0;256;[4, 5, 6, 7];[4, 1, 32]

### Allgather

Nodes broadcast their generated data to all other nodes. Therefore, some of the traffic is redundant, but it greatly simplifies index calculations and the whole communication can be done in two allgather calls (one for dispatch, one for combine).

The generated CSV files has a row for each allgather operation in the following format:

- Network data (B): Size of the tensor to be sent from a node in bytes
- Comm. group: A list of node ids that participate in allgather
- Dimensions: Tensor dimensions

For example, if node 4, 5, 6, 7 perform allgather for a tensor of [4,1,32] in fp16, the following row will be added to node4-7/decode*.csv:

*_ag;Allgather;0;0;0;256;[4, 5, 6, 7];[4, 1, 32]

### Alltoall

In dispatch stage, each node is responsible for sending a subset of samples. The nodes first calculate which sample should go to which node based on expert mapping and then perform the communication using alltoallv primitive.
In combine stage, each node is responsible for summing up the same subset of samples. Therefore, after expert calculations are done, each node calculates which output should go to which node and perform the communication again using alltoallv primitive.

The generated CSV files has a row for each alltoall operation in the following format:

- Network data (B): Size of the tensor to be sent from a node in bytes
- Comm. group: A list of node ids that participate in allgather
- Dimensions: Tensor dimensions, Send buffer split sizes, Receive buffer split sizes

For example, assume nodes 4, 5, 6, 7 perform alltoall with a communication matrix of [[2,5,0,1],[1,2,2,1],[0,3,2,2],[1,0,5,4]], where each number in the communication matrix represents the first dimension of a tensor of [None, 1, 32].

Then, in node4/decode*.csv:

*_a2a;AlltoAll;0;0;0;384;[4, 5, 6, 7];[8, 1, 32]

in node5/decode*.csv

*_a2a;AlltoAll;0;0;0;256;[4, 5, 6, 7];[6, 1, 32]

in node6/decode*.csv

*_a2a;AlltoAll;0;0;0;320;[4, 5, 6, 7];[7, 1, 32]

in node7/decode*.csv

*_a2a;AlltoAll;0;0;0;384;[4, 5, 6, 7];[10, 1, 32]


## Limitations
This repo is still an on-going project. If you require a new feature or encounter a bug, please create an issue.

Following system configurations are not supported:
* num_nodes should be equal to ```dp_attn * tp_attn * sp * pp``` **and** ```dp_ffn * tp_ffn * ep * pp```
* Currently, DP for FFN layers is not supported, so ```dp_ffn``` must be equal to 1.
* If ```ep > 1```, ```tp_ffn``` must be equal to 1. (you can still use DP or TP for attention layers)
* Batch size must be greater than or equal to ```dp_attn```

## License
BSD-3-Clause-Clear License

Copyright (c) 2024-2025 HUAWEI All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted (subject to the limitations in the disclaimer below) provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of HUAWEI nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE HUAWEI "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL HUAWEI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.