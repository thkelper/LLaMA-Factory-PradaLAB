#!/bin/bash
model_path=/root/autodl-tmp/models/llama-8b-instruct
model_name=L8BI
ft_name=dora_max1k
# dataset=commonsense_170k
dataset=code_80k
lr=9e-4
ptbs=2
pebs=1
gas=16
epoch=2.0
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
		--per_device_train_batch_size 2 \
		--gradient_accumulation_steps 16 \
		--lr_scheduler_type cosine \
		--logging_steps 10 \
		--learning_rate 9e-4 \
		--num_train_epochs 6 \
		--warmup_ratio 0.1 \
		--plot_loss \
		--fp16 \
		--report_to wandb \
		--wandb_project ${project} \
		--wandb_entity ${entity} \
		--max_samples 1000
		# --overwrite_output_dir \