from datasets import load_dataset, concatenate_datasets, DatasetDict
import datasets as ds
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

peft_model_id = input("Model Path(checkpoint): ")
model = AutoModelForCausalLM.from_pretrained(peft_model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(peft_model_id)

task = input("Task(commonsense or math): ")
output_path = input("Output Path: ")
if task == 'commonsense':
    dataset_names = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
elif task == 'math':
    dataset_names = ["MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq"]
else:
    raise ValueError("Invalid task provided. Please enter either 'commonsense' or 'math'.")

data_files = {name: f'/root/autodl-tmp/dataset/{name}/test.json' for name in dataset_names}
datasets = DatasetDict({name: load_dataset('json', data_files=data_files[name], split='train') for name in dataset_names})
combined_dataset = concatenate_datasets([ds for ds in datasets.values()])

def generate(instruction, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            # 使用tokenizer编码输入
            inputs = tokenizer(instruction, return_tensors="pt", max_length=1024, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=200)
            # 解码生成的文本
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
    return None

for dataset_name, dataset in datasets.items():
    for example in dataset:
        example['output'] = generate(example["instruction"])


datasets.save_to_disk(output_path)
