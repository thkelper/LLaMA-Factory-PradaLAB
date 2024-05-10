# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py

import inspect
import json
import os
from typing import Any, Dict, List, Optional

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

class PradaEvaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        logger.info("Loaded tokenizer")
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        logger.info("Loaded model")
        

if __name__ == "__main__":
    evaluator = PradaEvaluator()
