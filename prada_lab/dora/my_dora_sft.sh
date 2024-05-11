#!/bin/bash
model_path=/root/autodl-tmp/models/llama-8b-instruct
model_name=L8BI
ft_name=dora
dataset=commonsense_170k
# dataset=code_80k
lr=9e-4
ptbs=2
pebs=1
gas=16
epoch=6.0
project=DoRA_commonsense
entity=prada-lab
output_dir=/root/autodl-tmp/train_exps/${ft_name}_${model_name}_${dataset}_epoch${epoch}_lr${lr}_pbs${ptbs}_gas${gas}

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
	    --stage sft \
	    --do_train \
		--model_name_or_path ${model_path} \
		--dataset ${dataset} \
		--template default \
		--finetuning_type lora \
		--use_dora \
		--lora_target q_proj,v_proj \
		--output_dir ${output_dir} \
		--overwrite_cache \
		--per_device_train_batch_size ${ptbs} \
		--gradient_accumulation_steps ${gas} \
		--lr_scheduler_type cosine \
		--logging_steps 100 \
		--learning_rate ${lr} \
		--num_train_epochs ${epoch} \
		--warmup_ratio 0.1 \
		--plot_loss \
		--fp16 \
		--report_to wandb \
		--save_steps 3000 \
		--wandb_project ${project} \
		--wandb_entity ${entity} \
		# --max_samples 1000
		# --overwrite_output_dir \