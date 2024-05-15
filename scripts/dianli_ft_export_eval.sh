#!/bin/bash

method=neft8_Q8L # Quantization 8 Bit Lora
model=Qw14BC # Qwen 14B Chat
dataset=dianli_finetune_data_merge_1224_0425
dataset_name=dianliV2
lr=5e-4
epochs=6.0
pdbs=1
gas=1

exp_name=${method}_${model}_${dataset_name}_lr${lr}_ep${epochs}_pdbs${pdbs}_gas${gas}_4gpu
# Q8L_Qw14BC_dianliV1_lr5e-4_ep6.0_pdbs2_gas1_4gpu

output_dir=train_exps_env/${exp_name}
export_dir=export_dir/${method}_$(date +%m%d)
eval_save_dir=eval_results/${exp_name}


NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --include localhost:0,1,2,3,4,5,6,7 src/train_bash.py \
   --deepspeed scripts/ds2_config.json \
   --stage sft \
   --do_train \
   --model_name_or_path /mnt/data/models/Qwen/Qwen-14B-Chat-Int8 \
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
   --eval_steps 1000 \
   --evaluation_strategy steps \
   --learning_rate ${lr} \
   --num_train_epochs ${epochs} \
   --neftune_noise_alpha 5 \
   --wandb_project DianLiV2_FineTune
   
if [$? != 0]; then
    exit 1
fi

python3 src/export_model.py \
    --model_name_or_path /mnt/data/models/Qwen/Qwen-14B-Chat-Int8 \
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
    mv export_dir/${method}_$(date +%m%d) export_dir/${exp_name}
fi

python3 scripts/cal_score.py --folder ${eval_save_dir}