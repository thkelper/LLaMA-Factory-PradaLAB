#!/bin/bash
# exp_name=dianliV2_qwen_14b_chat_qlora_int8_lr5e-4_epochs6.0_ptbs4_gas4_lora_export
# export_dir=/app/LLaMA-Factory/export_dir/${exp_name}
# eval_save_dir=/app/LLaMA-Factory/eval_results/${exp_name}
export HF_DATASETS_CACHE=/app/.cache
export HF_CACHE_DIR=/app/.cache
export HF_HOME=/app/.cache/huggingface

export_dir=/root/autodl-tmp/export_exps/llamapro_L8BI_math_10k_epoch2.0_lr9e-4_pbs2_ga16
eval_save_dir=/root/autodl-tmp/eval_results/llamapro_L8BI_math_10k_epoch2.0_lr9e-4_pbs2_ga16

function evaluate(){
    model_path=${1}
    save_dir=${2}
    CUDA_VISIBLE_DEVICES=3 python3 src/prada_evaluate.py \
        --model_name_or_path ${model_path} \
        --template qwen \
        --lang zh\
        --task ceval \
        --split validation \
        --batch_size 1 \
        --save_dir ${save_dir} 

}

evaluate ${export_dir} \
        ${eval_save_dir}