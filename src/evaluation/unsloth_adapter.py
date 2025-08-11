import os
import logging
from unsloth import FastLanguageModel
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
import json 
import numpy as np
import torch
from src.core.loggers.interfaces.Ilogger import ILogger
from src.core.loggers.terminal_logger import TerminalLogger

class UnslothLM(HFLM):
    """Wrapper for Unsloth models to work with lm_eval"""
    def __init__(self, model, tokenizer, max_length=2048, batch_size=8):
        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size,
        )
        self._model = model
        self._tokenizer = tokenizer
        self.recent_result=None

class UnslothAdapter:
    def __init__(self,logger:ILogger=None):
        self.logger=logger or TerminalLogger()
    
    def load(self, model_path:str,max_length=2048,dtype="float16",batch_size=8,load_in_4bit=True):
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
        self.unsloth_lm=UnslothLM(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            batch_size=batch_size)
        self.model=model
        self.tokenizer=tokenizer
        
        return model, tokenizer
    def run_evaluate(self,tasks:list,num_fewshot=5,limit=None,random_seed=42):
        results = lm_eval.simple_evaluate(
            model=self.unsloth_lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            task_manager=TaskManager(include_path=None),
            limit=limit,
             random_seed=random_seed, 
        )
        return results