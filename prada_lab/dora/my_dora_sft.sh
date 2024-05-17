#!/bin/bash
model_path=/root/autodl-tmp/models/llama-8b-instruct
model_name=L8BI
ft_name=dora
# dataset=commonsense_170k
dataset=math_10k
# dataset=code_80k
lr=9e-4
ptbs=4
pebs=1
gas=4
epoch=2.0
project=DoRA_commonsense
entity=prada-lab
output_dir=/root/autodl-tmp/train_exps/${ft_name}_${model_name}_${dataset}_epoch${epoch}_lr${lr}_pbs${ptbs}_gas${gas}

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 deepspeed --num_gpus 2 src/train_bash.py \
		--deepspeed /root/LLaMA-Factory-PradaLAB/prada_lab/dora/ds_z2_config.json \
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
		--logging_steps 10 \
		--learning_rate ${lr} \
		--num_train_epochs ${epoch} \
		--warmup_ratio 0.1 \
		--plot_loss \
		--fp16 \
		--report_to wandb \
		--save_steps 200 \
		--wandb_project ${project} \
		--wandb_entity ${entity} \
		# --max_samples 1000
		# --overwrite_output_dir \