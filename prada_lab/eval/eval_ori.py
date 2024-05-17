"""
Dora:
python eval.py --task commonsense \
--ckpt_dir /root/autodl-tmp/fine-tuned_models/dora_C_set1 \
--base_model_path /root/autodl-tmp/llama3-8B \
--method dora

Pro:
python eval.py --task commonsense \
--ckpt_dir /root/autodl-tmp/fine-tuned_models/pro_C_set1 \
--base_model_path None \
--method pro
"""


import os
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re
import json
import csv


def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
        
def extract_answer_number(sentence: str) -> float:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py
    """
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def extract_answer_letter(sentence: str) -> str:
    """
    To ensure a fair comparison, we follow:
    https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/evaluate.py

    Note that it becomes ambiguous whether to extract the
    first letter or the last letter. Either way may lead
    to inaccurately assess the model performance. 

    We choose to follow the LLM-Adaptor repo, but leave this note
    for future research to explore the impact of this.
    """
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E', sentence_)
    if pred_answers:
        if not pred_answers:
            return ''
        return pred_answers[0]
    else:
        return ''
        
        
def extract_output(pred, trigger=''):
    if not trigger:
        return pred
    # for causallm only, use special trigger to detect new tokens.
    # if cannot find trigger --> generation is too long; default to empty generation
    start = pred.find(trigger)
    if start < 0:
        return ''
    output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
    return output
    



def main(ckpt_dir, base_model_path, task, method):
    
    if method == 'pro':
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16,
        )
        
    elif method == 'dora':    
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(model, ckpt_dir)
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side='left')
    
    model = model.to(device)
    
    if task == 'commonsense':
        max_new_tokens = 32
        dataset_names = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
        trigger_token = "the correct answer is "
    elif task == 'math':
        max_new_tokens = 512
        dataset_names = ["MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq"]
        trigger_token = "Assistant:"
    else:
        raise ValueError("Invalid task provided. Please enter either 'commonsense' or 'math'.")
    
    
    for name in dataset_names:
        # Load dataset
        data_file = f'/root/autodl-tmp/dataset/{name}/test.json'
        output_file = f'/root/autodl-tmp/evaluation/results/{method}_{task}/{name}.json'
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        dataset = load_dataset('json', data_files={'train': data_file}, split='train')
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        correct_count = 0
        total_count = 0
        results = []
        scores = {}
        for batch in tqdm(loader, desc=f"Processing {name}"):
            # Prepare batch inputs            
            instructions = [f"Human: {ins.strip()}\nAssistant: " for ins in batch['instruction']]


            inputs = tokenizer(instructions, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs_ids = inputs['input_ids']
                
            with torch.no_grad():
                outputs = model.generate(
                    inputs_ids, 
                    max_new_tokens=max_new_tokens, 
                    pad_token_id=tokenizer.eos_token_id, 
                    do_sample=False,
                )

            actual_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # print(actual_preds)
            # print('--=-=-=')
            
            # Evaluate predictions
            for idx, pred in enumerate(actual_preds):
                raw_generation = extract_output(pred, trigger_token)
                # import pdb
                # pdb.set_trace()
                
                answer = batch["answer"][idx].strip()
                if task == "commonsense":
                    generation = raw_generation[:]
                    if generation.strip() == answer.strip():
                        correct_count += 1
                elif task == "math":
                    if not is_float(answer): # assuming this is from AQuA:
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        generation = extract_answer_number(raw_generation)
                        if abs(float(answer) - generation) <= 0.001:
                            correct_count += 1
                results.append({"question": instructions[idx], "generated_answer": generation, "actual_answer": answer})
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=4)
                total_count += 1

        
        score = f"{name}, {correct_count / total_count:.3f}"
        print(score)
        scores[name] = score
    with open(f'/root/autodl-tmp/evaluation/results/{method}_{task}_score.csv', mode='w', newline='', encoding='utf-8') as file:
        fields = ['dataset', 'score']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in dataset_names:
            writer.writerow({'dataset': name, 'score': scores[name]})



        

if __name__ == "__main__":
    fire.Fire(main)