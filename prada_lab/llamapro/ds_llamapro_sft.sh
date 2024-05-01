#!/bin/bash
model_path=/root/autodl-tmp/models/llama-8b-instruct-pro
model_name=L8BI
ft_name=llamapro
dataset=math_10k
lr=5e-5
pbs=2
ga=16
epoch=2.0
project=llama\ pro
entity=prada-lab

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 1 src/train_bash.py \
    --deepspeed /root/LLaMA-Factory-PradaLAB/prada_lab/llamapro/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path ${model_path} \
    --dataset ${dataset} \
    --template default \
    --finetuning_type freeze \
    --name_module_trainable all \
    --num_layer_trainable 4 \
    --use_llama_pro \
    --output_dir /root/autodl-tmp/train_exps/${ft_name}_${model_name}_${dataset}_epoch${epoch}_lr${lr}_pbs${pbs}_ga${ga} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 1024 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size ${pbs} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${ga} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate ${lr} \
    --num_train_epochs ${epoch} \
    --max_samples 3000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16 \
    # --wandb_project ${project} \
    # --wandb_entity ${entity}
