#!/bin/bash

method=neft5_Q8L # Quantization 8 Bit Lora
model=Qw14BC # Qwen 14B Chat
dataset=dianli_finetune_data_merge_1224_0425
dataset_name=dianliV2
lr=5e-4
epochs=4.0
pdbs=2
gas=1
gpu_ids=0,1,2,3
gpu_nums=$(( $(echo $gpu_ids | tr -cd ',' | wc -c) + 1 ))

exp_name=${method}_${model}_${dataset_name}_lr${lr}_ep${epochs}_pdbs${pdbs}_gas${gas}_${gpu_nums}gpu

output_dir=train_exps/${exp_name}
export_dir=export_dir/${method}_$(date +%m%d)
eval_save_dir=eval_results/${exp_name}
base_model_path=/mnt/data/models/Qwen/Qwen-14B-Chat

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --include localhost:${gpu_ids} src/train_bash.py \
   --deepspeed scripts/ds2_config.json \
   --stage sft \
   --do_train \
   --model_name_or_path ${base_model_path} \
   --dataset ${dataset} \
   --template qwen \
   --finetuning_type lora \
   --quantization_bit 8 \
   --lora_rank 8 \
   --lora_dropout 0.2 \
   --lora_alpha 16 \
   --lora_target c_attn \
   --output_dir ${output_dir} \
   --overwrite_cache \
   --overwrite_output_dir \
   --cutoff_len 1024 \
   --preprocessing_num_workers 16 \
   --per_device_train_batch_size ${pdbs} \
   --per_device_eval_batch_size 2 \
   --gradient_accumulation_steps ${gas} \
   --lr_scheduler_type cosine \
   --logging_steps 100 \
   --warmup_steps 50 \
   --save_steps 1000 \
   --eval_steps 500 \
   --val_size 0.01 \
   --evaluation_strategy steps \
   --learning_rate ${lr} \
   --num_train_epochs ${epochs} \
   --wandb_project dianliV2 \
   --neftune_noise_alpha 5 \
   
if [$? != 0]; then
    exit 1
else
    # to ensure the reproducibility, copy finetune scripts to outputdir
    self_path=$0
    cp_self_to="$output_dir/$(basename $self_path)"
    cp ${self_path} ${cp_self_to}
    if [[ -f "$cp_self_to" ]]; then
        echo "Finetune Script copied successfully to $cp_self_to"
    else
        echo "Failed to copy the script."
        exit 1
    fi
fi

python3 src/export_model.py \
    --model_name_or_path ${base_model_path} \
    --template qwen \
    --finetuning_type lora \
    --adapter_name_or_path ${output_dir} \
    --export_dir ${export_dir}


function evaluate(){
    model_path=${1}
    save_dir=${2}
    CUDA_VISIBLE_DEVICES=0 python3 src/evaluate.py \
        --model_name_or_path ${model_path} \
        --template qwen \
        --lang zh\
        --task ceval \
        --split validation \
        --batch_size 1 \
        --quantization_bit 8 \
        --save_dir ${save_dir} 
        # --n_shot 0
}

evaluate ${export_dir} \
        ${eval_save_dir}

if [ $? != 0 ];then
    exit
else
    mv ${export_dir} export_dir/${exp_name}
fi

python3 scripts/cal_score.py --folder ${eval_save_dir}
