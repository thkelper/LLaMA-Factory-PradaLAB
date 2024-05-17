# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import inspect
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template
from loguru import logger
from torch.utils.data import DataLoader
from ..data import Role
"""
python eval.py --task commonsense \
--ckpt_dir /root/autodl-tmp/fine-tuned_models/dora_C_set1 \
--base_model_path /root/autodl-tmp/llama3-8B \
--method dora
"""



from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader
import torch
import fire
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import re
import json


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


def format_example(
    self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str
) -> List[Dict[str, str]]:
    r"""
    Converts dataset examples to messages.
    """
    messages = []
    for k in range(len(support_set)):
        prompt, response = self._parse_example(support_set[k])
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})

    prompt, response = self._parse_example(target_data)
    messages.append({"role": Role.USER.value, "content": prompt})
    messages.append({"role": Role.ASSISTANT.value, "content": response})
    messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
    return messages

class PradaEvaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        logger.info("Loaded tokenizer")
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)


    def eval():
        if task == 'commonsense':
            dataset_names = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
            trigger_token = "the correct answer is "
        elif task == 'math':
            dataset_names = ["MultiArith", "gsm8k", "SVAMP", "mawps", "AddSub", "AQuA", "SingleEq"]
            trigger_token = "The answer is "
        else:
            raise ValueError("Invalid task provided. Please enter either 'commonsense' or 'math'.")

        for data_name in dataset_names:
            data_file = f'/root/autodl-tmp/dataset/{data_name}/test.json'   
            dataset = load_dataset('json', data_files={'train':data_file}, split='train')
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            correct_count = 0
            total_count = 0

            for batch in tqdm(loader, desc=f"Processing {data_name}"):
                one_shot_template = ""
                # instruction, output, answer = batch['instruction'][0]
                instruction, output, answer = batch['instruction'], batch['output'], batch['answer']
                question = output.replace(answer, "")
                instruction = instruction + "，" + question
                inputs = None

if __name__ == "__main__":
    evaluator = PradaEvaluator()
