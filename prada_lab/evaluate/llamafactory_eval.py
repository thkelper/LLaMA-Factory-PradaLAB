from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re
import json
import os.path as osp
import regex


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
        
        
# def extract_output(pred, trigger=''):
#     if not trigger:
#         return pred
#     # for causallm only, use special trigger to detect new tokens.
#     # if cannot find trigger --> generation is too long; default to empty generation
#     start = pred.find(trigger)
#     if start < 0:
#         return 
#     output = pred[start+len(trigger):].lstrip() # left strip any whitespaces
#     return output

# def extract_output(pred):

    

def main(ckpt_dir, task, method, base_model_path=None, debug=False):
    if method == 'pro':
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.float16,
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side = "left")
        model = model.to(device)
        
    elif method == 'dora':    
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
        )
        model.generation_config.temperature=None
        model.generation_config.top_p=None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, padding_side = "left")
        model = PeftModel.from_pretrained(model, ckpt_dir)
        model = model.to(device)
    
    if task == 'commonsense':
        dataset_names = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
        trigger_token = "the correct answer is "
    elif task == 'math':
        dataset_names = ["MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq"]
        # dataset_names = ['MultiArith']
        trigger_token = "The answer is "
        # answer_template = "Response Format is {The answer is {your answer}}."
    else:
        raise ValueError("Invalid task provided. Please enter either 'commonsense' or 'math'.")
    
    for name in dataset_names:
        # Load dataset
        
        data_root = '/root/autodl-tmp/datasets'
        # data_file = f'data_root/{name}/test.json'
        data_path = osp.join(data_root, name, 'test.json')
        if debug:
            data_fname = osp.basename(data_path)
            ftitle, fext = osp.splitext(data_fname)
            new_data_fname = f'{name}_test_size{fext}'
            new_data_path = osp.join(data_root, name, new_data_fname)
            with open(data_path, 'r') as fp:
                data_list = json.load(fp)
                
                # test_size = int( * sample)
                test_size = min(len(data_list), 100)
                test_data_list = data_list[:test_size]
            with open(new_data_path, 'w') as fp:
                json.dump(test_data_list, fp)
            dataset = load_dataset('json', data_files={'train': new_data_path}, split='train')
        else:
            dataset = load_dataset('json', data_files={'train': data_path}, split='train')

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        correct_count = 0
        total_count = 0
        if debug:
            debug_result = list()

        for batch in tqdm(loader, desc=f"Processing {name}"):
            # Prepare batch inputs
            #  instruction = "Question: " + batch["instruction"][0] + "the correct answer is "
            # instruction = batch["instruction"][0] + ". the correct answer is "
            import pdb
            pdb.set_trace()
            instruction, output, answer = batch['instruction'][0], batch['output'][0], batch['answer'][0]
            # import pdb
            # pdb.set_trace()
            # if answer in output
            # question = output.replace(answer, "")
            # instruction = instruction + "," + question
            # instruction = instruction + ',' + trigger_token
            # instruction = instruction + ',' + answer_template

            inputs = tokenizer(instruction, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs_ids = inputs['input_ids']

            
            with torch.no_grad():
                outputs = model.generate(inputs_ids, max_length=512, pad_token_id=tokenizer.eos_token_id, do_sample=False)

            actual_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # import pdb
            # pdb.set_trace()
            # Evaluate predictions
            for idx, pred in enumerate(actual_preds):
                
                # raw_generation = extract_output(pred, trigger_token)
                answer = batch["answer"][idx].strip()
                if task == "commonsense":
                    # generation = raw_generation[:]
                    raw_generation = regex.split(r'\n+', pred)[-1]
                    # import pdb
                    # pdb.set_trace()
                    if generation.strip() == answer.strip():
                        correct_count += 1
                elif task == "math":
                    # answer = eval(answer)
                    raw_generation = regex.split(r'\n+', pred)[-1]
                    if not is_float(answer): # assuming this is from AQuA:
                        generation = extract_answer_letter(raw_generation)
                        if generation.strip() == answer.strip():
                            correct_count += 1
                    else:
                        generation = extract_answer_number(raw_generation)
                        if abs(float(answer) - generation) <= 0.001:
                            correct_count += 1
                total_count += 1
                if debug:
                    debug_result.append(dict(instruction=instruction, answer=answer, pred=actual_preds, pred_answer=generation))
                    
        
        accuracy = f'{correct_count / total_count:.3f}'
        if debug:
            new_debug_fname = f'{name}_debug_{accuracy}{fext}'
            new_debug_path = osp.join(data_root, name, new_debug_fname)
            with open(new_debug_path, "w") as fp:
                json.dump(debug_result, fp, indent=2)
        
                

        # Print evaluation result for the current dataset
        print(f"Accuracy for {name}: {correct_count / total_count:.3f}")


if __name__ == "__main__":
    fire.Fire(main)