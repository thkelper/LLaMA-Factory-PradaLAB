#!/bin/bash

python scripts/llama_pro.py \
    --model_name_or_path /root/autodl-tmp/models/llama-8b-instruct \
    --output_dir /root/autodl-tmp/models/llama-8b-instruct-pro \
    --num_expand 1
