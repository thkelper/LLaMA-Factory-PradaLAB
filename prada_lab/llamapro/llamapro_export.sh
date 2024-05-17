output_dir=/root/autodl-tmp/train_exps/llamapro_L8BI_commonsense_170k_epoch6.0_lr9e-4_pbs4_ga16
export_dir=/root/autodl-tmp/export_exps/llamapro_L8BI_commonsense_170k_epoch6.0_lr9e-4_pbs4_ga16

python3 src/export_model.py \
    --model_name_or_path ${output_dir} \
    --template llama3 \
    --export_dir ${export_dir} \
    --finetuning_type freeze
    # --adapter_name_or_path ${output_dir} \