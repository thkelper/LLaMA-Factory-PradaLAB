from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def generate(instruction, tokenizer, model, device, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            inputs = tokenizer(instruction, return_tensors="pt", max_length=512, truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=200)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Attempt {retries + 1} failed: {e}")
            retries += 1
    return 'Error'



def main(ckpt_dir, base_model_path, output_path, task):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = PeftModel.from_pretrained(model, ckpt_dir)
    
    if task == 'commonsense':
        dataset_names = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
    elif task == 'math':
        dataset_names = ["MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq"]
    else:
        raise ValueError("Invalid task provided. Please enter either 'commonsense' or 'math'.")
    
    data_files = {name: f'/root/autodl-tmp/dataset/{name}/test.json' for name in dataset_names}

    dataset_files = {name: load_dataset('json', data_files=data_files[name], split='train') for name in dataset_names}
    combined_dataset = concatenate_datasets([ds for ds in dataset_files.values()])
    for example in tqdm(combined_dataset, desc="Generating outputs"): 
        example["output"] = generate(example["instruction"], tokenizer, model, device)
    combined_dataset.to_json(output_path)
    
if __name__ == "__main__":
    fire.Fire(main)