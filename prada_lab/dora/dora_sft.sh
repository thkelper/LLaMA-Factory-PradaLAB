model_path=/root/autodl-tmp/models/llama-8b-instruct-pro
model_name=L8BI
ft_name=dora
dataset=math_10k
lr=5e-5
pbs=2
ga=16
epoch=2.0
project=DoRA_commonsense
entity=prada-lab

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
	    --stage sft \
	    --do_train \
		--model_name_or_path ${model_path} \
		--dataset ${dataset} \
		--template default \
		--finetuning_type lora \
		--use_dora \
		--lora_target q_proj,v_proj \
		--output_dir /root/autodl-tmp/train_exps/dora_llama-8b-instruct_commonsense_lr9e-4_pbs2_ga16 \
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
		--report_to wandb
		# --overwrite_output_dir \
