import wandb
import torch 
import numpy as np
import pandas as pd 
from tqdm.std import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt 

import datasets
from transformers import PreTrainedTokenizerFast
from transformers import RobertaForSequenceClassification

import os
import sys
import tempfile
import warnings
import contextlib

mpl.rcParams["mathtext.fontset"] = "cm"
warnings.filterwarnings('ignore')
api = wandb.Api() 

dataset = datasets.load_from_disk('../data/raw')

df = pd.read_csv("../malberta-runs.csv")
df = df[df['State'] == "finished"]
pretrained = df[df['Tags'].str.contains("pretrain-experiment", regex=False, na=False, case=False)]
base = df[~df['Tags'].str.contains("pretrain", regex=False, na=False, case=False)]

def tokenize(model, sample, tokenizer):
    seq_len = model.config.max_position_embeddings - 2

    input = tokenizer(
        sample['text'],
        padding='max_length',
        max_length=seq_len,
        return_overflowing_tokens=True,
        truncation=True,
        return_special_tokens_mask=True,
        return_tensors='pt'
    )

    return input

def get_embeddings(model, sample, tokenizer, max_len = None):
    input = tokenize(model, sample, tokenizer)
    
    torch.mps.empty_cache()

    with torch.no_grad():
        return model.get_input_embeddings()(input['input_ids'][:max_len])

def get_attention(model, sample, tokenizer, max_len = None):
    input = tokenize(model, sample, tokenizer)

    torch.mps.empty_cache()

    with torch.no_grad():
        return model(input['input_ids'][:max_len], input['attention_mask'][:max_len], output_attentions=True).attentions

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout

def get_model(id):
    other = pretrained[pretrained['Notes'] == f'"Pretrained {id}"']

    if other.empty: 
        raise Exception("Could not find a model with that ID.")

    with tempfile.TemporaryDirectory() as tdir: 
        base_artifact = api.artifact(f'henry-williams/opcode-malberta/model-{id}:v1', type='model')
        pretrained_artifact = api.artifact(f'henry-williams/opcode-malberta/model-{other.ID.values[0]}:v1', type='model')

        with suppress_stdout():
            pretrained_path = pretrained_artifact.download(root=os.path.join(tdir, "pretrained"))
            base_path = base_artifact.download(root=os.path.join(tdir, 'base'))

        return PreTrainedTokenizerFast.from_pretrained(base_path), RobertaForSequenceClassification.from_pretrained(base_path), RobertaForSequenceClassification.from_pretrained(pretrained_path)