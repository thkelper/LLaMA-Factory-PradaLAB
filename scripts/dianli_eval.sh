exp_name=neft5_Q8L_Qw14BC_dianliV2_lr5e-4_ep4.0_pdbs2_gas1_4gpu
n_shot=0
export_dir=export_dir/${exp_name}
export_dir_simple=export_dir/n_shot${n_shot}_$(date +%m%d)
mv ${export_dir} ${export_dir_simple}
# eval_save_dir=eval_results/${exp_name}

eval_save_dir=eval_results/neft5_Q8L_Qw14BC_dianliV2_lr5e-4_ep4.0_pdbs2_gas1_4gpu_nshot${n_shot}
gpu_ids=7
function evaluate(){
    model_path=${1}
    save_dir=${2}
    CUDA_VISIBLE_DEVICES=${gpu_ids} python3 src/evaluate.py \
        --model_name_or_path ${model_path} \
        --template qwen \
        --lang zh\
        --task ceval \
        --split validation \
        --batch_size 1 \
        --quantization_bit 8 \
        --save_dir ${save_dir} \
        --n_shot ${n_shot}
}

evaluate ${export_dir_simple} \
        ${eval_save_dir}


if [$? != 0]; then
    exit 1
else
    # to ensure the reproducibility, copy eval scripts to outputdir
    self_path=$0
    cp_self_to="$eval_save_dir/$(basename $self_path)"
    cp ${self_path} ${cp_self_to}
    if [[ -f "$cp_self_to" ]]; then
        echo "Finetune Script copied successfully to $cp_self_to"
    else
        echo "Failed to copy the script."
        exit 1
    fi
fi
python3 scripts/cal_score.py --folder ${eval_save_dir}

if [$? == 0]; then
    mv ${export_dir_simple} ${export_dir}