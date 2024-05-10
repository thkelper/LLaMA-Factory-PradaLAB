#!/bin/bash
model_path=/root/autodl-tmp/models/llama-8b-instruct-pro
model_name=L8BI
ft_name=llamapro
dataset=commonsense_170k
lr=9e-4
ptbs=4
pebs=1
gas=16
epoch=6.0
project=llamapro
entity=prada-lab
output_dir=/root/autodl-tmp/train_exps/${ft_name}_${model_name}_${dataset}_epoch${epoch}_lr${lr}_pbs${ptbs}_ga${gas}


NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 2 src/train_bash.py \
    --deepspeed /root/LLaMA-Factory-PradaLAB/prada_lab/llamapro/ds_z2_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path ${model_path} \
    --dataset ${dataset} \
    --template default \
    --finetuning_type freeze \
    --name_module_trainable all \
    --num_layer_trainable 1 \
    --use_llama_pro \
    --output_dir ${output_dir} \
    --overwrite_cache \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${ptbs} \
    --per_device_eval_batch_size ${pebs}1 \
    --gradient_accumulation_steps ${gas} \
    --overwrite_output_dir \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --warmup_steps 20 \
    --load_best_model_at_end \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --val_size 0.01 \
    --plot_loss \
    --fp16 \
    --wandb_project ${project} \
    --wandb_entity ${entity} \
    --save_steps 1000 \
    --evaluation_strategy steps \
    # --max_samples 1000
    # --eval_steps 100 \
    
    
