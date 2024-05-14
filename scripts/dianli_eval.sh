exp_name=Q8L_Qw14BC_dianliV1_lr5e-4_ep6.0_pdbs2_gas1_4gpu
export_dir=export_dir/Q8L_0514
eval_save_dir=eval_results/${exp_name}

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

# evaluate ${export_dir} \
#         ${eval_save_dir}

# if [ $? != 0 ];then
#     exit
# else
#     mv ${export_dir} export_dir/exp_name
# fi

# mv ${export_dir} export_dir/exp_name
python3 scripts/cal_score.py --folder ${eval_save_dir}
