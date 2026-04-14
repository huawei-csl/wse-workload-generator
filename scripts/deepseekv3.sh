#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Generate node level partitions and compute graph for the specified layers of DeepSeekV3 model
python src/generate_nodes.py \
    --model_config configs/deepseekv3.json \
    --system_config configs/system.json \
    --bsz 1024 \
    --prefill_len 2048 \
    --decode_len 10 \
    --only_decode 1 \
    --simplified_decode 1 \
    --nodes "all" \
    --layers "decode10" \
    --dtype fp16 \
    --log "info" \
    --outdir "./output"
    
# Generate core level traces for the specified layers
python src/generate_traces.py \
    --system_config configs/system.json \
    --layers "decode10" \
    --iter "decode0" \
    --dtype fp16 \
    --log "info" \
    --outdir "./output"

# Visualize the generated traces
python src/visualize_traces.py \
    --system_config configs/system.json \
    --layers "decode10" \
    --iter "decode0" \
    --log "info" \
    --outdir "./output"