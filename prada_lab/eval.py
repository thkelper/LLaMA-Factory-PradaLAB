import os
import torch
import argparse
from transformers import (
    AutoConfig,
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
    TrainingArguments
)
import wandb
import datetime
import json
import math
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import Dataset
from peft import PeftModel
from task_config import task_config
from dataset import LoReftGLUEDataset, LoReftSupervisedDataset
from compute_metrics import compute_metrics

from pyreft import ReftDataCollator

device = "cuda" if torch.cuda.is_available() else "cpu"
classification_tasks = {"glue"}
dtype_mapping = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float8": "float8",
}



def evaluation(
    act_fn: str,
    add_bias: bool,
    model: str,
    layers: str,
    rank: int,
    position: str,
    epochs: int,
    seed: int,
    max_n_train_example: int,
    max_n_eval_example: int,
    is_wandb: bool,
    wandb_name: str,
    gradient_accumulation_steps: int,
    batch_size: int,
    output_dir: str,
    task: str,
    lr: float,
    schedule: str,
    data_dir: str,
    train_dataset: str,
    eval_dataset: str,
    save_model: bool,
    eval_batch_size: int,
    warmup_ratio: float,
    weight_decay: float,
    dropout: float,
    test_split: str,
    train_on_inputs: bool,
    max_length: int,
    use_normalized_template: bool,
    allow_cls_grad: bool,
    metric_for_best_model: str,
    dtype: str,
    logging_steps: int,
    wandb_dir: str,
    wandb_proj: str,
    share_weights: bool,
    greedy_decoding: bool,
    temperature: float,
    top_p: float,
    top_k: float,
    args,
    ckpt_dir,
):
    """
    Generic Representation Finetuning.
    """

    assert task in {
        "commonsense", "math", "alpaca", "instruct", "ultrafeedback", "glue", "gsm8k",
        "ultrafeedback_pair"
    }

    dtype = dtype_mapping[dtype]
    
    # store/log run details
    print(
        f"task: {task}, model: {model}"
        f"max_length: {max_length}, allow_cls_grad: {allow_cls_grad}"
    )

    # everything is guarded by a single seed
    set_seed(seed)

    model_name = model
    model_str = model.split("/")[-1]
    train_dataset_str = train_dataset
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    if train_dataset is not None:
        run_name = f"{model_str}.{task}.{train_dataset_str}.{test_split}.{now}"
    else:
        run_name = f"{model_str}.{task}.{now}"
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_dir,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.unk_token == None and tokenizer.pad_token == None:
        # raw llama3
        print("adding a special padding token...")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        need_resize = True
    else:
        tokenizer.pad_token = tokenizer.unk_token
        need_resize = False


    # load dataset splits
    assert task in task_config, f"Unrecognized task: {task}"
    train_datasets = task_config[task]["train_datasets"] if train_dataset is None else [train_dataset]
    eval_datasets = task_config[task]["eval_datasets"] if eval_dataset is None else [eval_dataset]
        
    ReftDataset = LoReftGLUEDataset if task == "glue" else LoReftSupervisedDataset 
    train_dataset = ReftDataset(
        task, train_datasets[0] if task == "glue" or task == "ultrafeedback_pair" \
            else (os.path.join(data_dir, train_datasets[0]) if data_dir is not None else train_datasets[0]), 
        tokenizer, data_split="train", seed=seed, max_n_example=max_n_train_example,
        **{"num_interventions": len(layers), "position": position, 
           "share_weights": share_weights, "test_split": test_split}
    )
    trigger_tokens = train_dataset.trigger_tokens
    num_labels = train_dataset.num_labels

    all_eval_datasets = {}
    for eval_dataset in eval_datasets:
        test_splits = test_split.split(";")
        all_eval_datasets[eval_dataset] = {}
        for split in test_splits:
            raw_eval = ReftDataset(
                task, eval_dataset if task == "glue" else os.path.join(data_dir, eval_dataset), 
                tokenizer, data_split=split, seed=seed, max_n_example=max_n_eval_example,
                **{"num_interventions": len(layers), "position": position, 
                   "share_weights": share_weights}
            )
            all_eval_datasets[eval_dataset][split] = [raw_eval, raw_eval.raw_dataset]
    eval_datasets = all_eval_datasets


    
    # select collator based on the type
    if task in classification_tasks:
        data_collator_fn = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="longest"
        )
    else:
        data_collator_fn = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding="longest"
        )
    data_collator = ReftDataCollator(data_collator=data_collator_fn)

    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, ckpt_dir)
    model.print_trainble_parameters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_params = model.count_parameters(include_model=False)





    # do evaluate
    model.eval()
    
    eval_results = {}
    for dataset_name in eval_datasets:
        # split evalset into chunks
        for split, (eval_dataset, data_items) in eval_datasets[dataset_name].items():
            
            generations, stats = compute_metrics(
                task, dataset_name, model, tokenizer, eval_dataset, data_items,
                trigger_tokens, run_name, eval_batch_size, 
                data_collator if task in classification_tasks else None,
                split, temperature, top_p, top_k
            )

            # log
            eval_results.update(stats)
            if is_wandb:
                wandb.log(stats)
            generations = stats if generations is None else generations
            print(stats)
            result_json_file_name = f"{output_dir}/{run_name}/{dataset_name}_{split}_outputs.json"
            with open(result_json_file_name, 'w') as json_file:
                json.dump(generations, json_file, indent=4)

    # log final eval stats
    result_json_file_name = f"{output_dir}/{run_name}/eval_results.json"
    eval_results["n_params"] = n_params
    with open(result_json_file_name, 'w') as json_file:
        json.dump(eval_results, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="A simple script that takes different arguments.")
    
    parser.add_argument('-task', '--task', type=str, default=None)
    parser.add_argument('-data_dir', '--data_dir', type=str, default="./datasets")
    parser.add_argument('-train_dataset', '--train_dataset', type=str, default=None)
    parser.add_argument('-eval_dataset', '--eval_dataset', type=str, default=None)
    parser.add_argument('-model', '--model', type=str, help='yahma/llama-7b-hf', default='yahma/llama-7b-hf')
    parser.add_argument('-seed', '--seed', type=int, help='42', default=42)
    parser.add_argument('-l', '--layers', type=str, help='2;10;18;26', default='2;10;18;26')
    parser.add_argument('-r', '--rank', type=int, help=8, default=8)
    parser.add_argument('-p', '--position', type=str, help='f1+l1', default='f1+l1')
    parser.add_argument('-e', '--epochs', type=int, help='1', default=1)
    parser.add_argument('-ckpt_dir', '--ckpt_dir', type=str, default='./checkpoint')
    parser.add_argument('-is_wandb', '--is_wandb', action='store_true')
    parser.add_argument('-wandb_name', '--wandb_name', type=str, default="reft")
    parser.add_argument('-save_model', '--save_model', action='store_true')
    parser.add_argument('-max_n_train_example', '--max_n_train_example', type=int, default=None)
    parser.add_argument('-max_n_eval_example', '--max_n_eval_example', type=int, default=None)
    parser.add_argument('-gradient_accumulation_steps', '--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('-batch_size', '--batch_size', type=int, default=4)
    parser.add_argument('-eval_batch_size', '--eval_batch_size', type=int, default=4)
    parser.add_argument('-output_dir', '--output_dir', type=str, default="./official_results")
    parser.add_argument('-lr', '--lr', type=float, default=5e-3)
    parser.add_argument('-schedule', '--schedule', type=str, default='linear')
    parser.add_argument('-wu', '--warmup_ratio', type=float, default=0.00)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.00)
    parser.add_argument('-dropout', '--dropout', type=float, default=0.00)
    parser.add_argument('-act_fn', '--act_fn', type=str, default=None)
    parser.add_argument('-add_bias', '--add_bias', action='store_true')
    parser.add_argument('-test_split', '--test_split', type=str, default="validation")
    parser.add_argument('-train_on_inputs', '--train_on_inputs', action='store_true')
    parser.add_argument('-max_length', '--max_length', type=int, help=512, default=512)
    parser.add_argument('-nt', '--use_normalized_template', action='store_true')
    parser.add_argument('-allow_cls_grad', '--allow_cls_grad', action='store_true')
    parser.add_argument('-metric_for_best_model', '--metric_for_best_model', type=str, default="accuracy")
    parser.add_argument('-dtype', '--dtype', type=str, default="bfloat16" if device == "cuda" else "float32")
    parser.add_argument('-logging_steps', '--logging_steps', type=int, help=1, default=1)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, default='wandb')
    parser.add_argument('-wandb_proj', '--wandb_proj', type=str, default='MyReFT')
    parser.add_argument('-sw', '--share_weights', action='store_true')
    parser.add_argument('-gd', '--greedy_decoding', action='store_true')

    # decoding params
    parser.add_argument('-t', '--temperature', type=float, default=None)
    parser.add_argument('-top_p', '--top_p', type=float, default=None)
    parser.add_argument('-top_k', '--top_k', type=float, default=None)

    args = parser.parse_args()

    evaluation(**vars(args), args=args)


if __name__ == "__main__":
    main()