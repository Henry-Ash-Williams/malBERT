import torch
import wandb 
import r2pipe 
import numpy as np 
from tqdm import trange, tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, PreTrainedTokenizerFast

import os
import json
import random
import tempfile
import subprocess
from os import PathLike

device = torch.device('mps')

MAL_PATH = "/Volumes/malware-dataset/unobfuscated-exes/"
BEN_PATH = "/Volumes/malware-dataset/benign_program_dataset_WinXP_SP3/benign_programs_WinXP_SP3"

def get_model(id):
    with tempfile.TemporaryDirectory() as tdir: 
        artifact = api.artifact(f'henry-williams/opcode-malberta/model-{id}:v1', type='model')
        base_path = artifact.download(root=tdir)

        return RobertaForSequenceClassification.from_pretrained(base_path)


def get_disassm(path: PathLike):
    r2 = r2pipe.open(path, ['-12'])
    r2.cmd("aaa")

    info = r2.cmdj("ij")

    if info["bin"]["arch"] != "x86":
        return []

    section_info = r2.cmdj("iSj")
    executable_sections = [
        section for section in section_info if "x" in section.get("perm", "")
    ]

    full_disassembly = []

    for section in executable_sections:
        start = section["vaddr"]
        size = section["vsize"]

        disassembly = r2.cmdj(f"pdaj {size} @ {start}")

        valid = [instr for instr in disassembly if set(instr["bytes"]) != {"0"}]
        full_disassembly.extend(valid)

    return full_disassembly


def make_opcode_sequences(instrs):
    valid_instrs = [
        instruction["inst"]
        for instruction in instrs
        if instruction["inst"] != "invalid"
    ]

    return ' '.join([instr.split(' ')[0] for instr in valid_instrs])

def tokenize(sample, model):
    seq_length = model.config.max_position_embeddings

    input = tokenizer(
        sample,
        padding='max_length',
        max_length=seq_length - 2,
        return_overflowing_tokens=True,
        truncation=True,
        return_special_tokens_mask=True,
        return_tensors='pt'
    )

    return input

def pipeline(model, path: os.PathLike, batch_size=64):
    if not os.path.exists(path):
        raise Exception(f"Could not find specified file at {path}")
    disassembly = get_disassm(path)
    opcodes = make_opcode_sequences(disassembly)
    input = tokenize(opcodes, model)

    model.eval()

    logits = []
    input_ids = input['input_ids'].split(batch_size)
    attention_mask = input['attention_mask'].split(batch_size)
    token_type_ids = input['token_type_ids'].split(batch_size)
    torch.mps.empty_cache()
    for ids, attn_mask, tok_ty_ids in tqdm(zip(input_ids, attention_mask, token_type_ids), total=len(input_ids), position=2, leave=False, desc=path.split('/')[-1]):
        ids = ids.to(device)
        attn_mask = attn_mask.to(device)
        tok_ty_ids = tok_ty_ids.to(device)

        with torch.no_grad():
            logits.append(model(
                input_ids=ids,
                attention_mask=attn_mask,
                token_type_ids=tok_ty_ids
            ))
    
    logits = torch.vstack([logit.logits for logit in logits])
    return F.softmax(logits.mean(dim=0), dim=0)


def experiment_step(model, file, p=1.0, batch_size=64):
    ''' 
    p is the likelihood of the file being obfuscated
    '''
    with tempfile.TemporaryDirectory() as tdir: 
        if random.random() < p: 
            obfuscated_path = os.path.join(tdir, file.split('/')[-1])
            subprocess.run(['upx', '-o', obfuscated_path, file], stdout=subprocess.DEVNULL, stderror=subprocess.DEVNULL)
            return pipeline(model, obfuscated_path, batch_size=batch_size)
        else: 
            return pipeline(model, file, batch_size=batch_size)


def run_experiment(files, labels, model, p=0.0, batch_size=256):
    predicted = []
    actual = []

    for file, label in tqdm(zip(files, labels), total=len(files), leave=False, desc=f'p = {p:.2}', position=1): 
        try: 
            logits = experiment_step(model, file, p=p, batch_size=batch_size)
        except: 
            continue 

        predicted.append(logits.argmax().item())
        actual.append(label)

    return classification_report(actual, predicted, output_dict=True)

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['WANDB_SILENT'] = "true"

    api = wandb.Api() 
    runs = {
        'frozen_pretrained': "t4o7wvla",
        'pretrained': "e1tosi4k",
        'base': "e96b8h5a"
    }
    model = get_model(runs['base']).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("/Users/henrywilliams/Documents/programming/python/ai/malbert-test/MalBERTa")
    obfuscation_likelihood = np.linspace(0, 1, 20)

    ben_files = [os.path.join(BEN_PATH, file) for file in os.listdir(BEN_PATH) if file.endswith('.exe') and not file.startswith('._')]
    mal_files = [os.path.join(MAL_PATH, file) for file in os.listdir(MAL_PATH) if file.endswith('.exe') and not file.startswith('._')]
    random.shuffle(ben_files)
    ben_files = ben_files[:len(mal_files)]

    files = mal_files + ben_files 
    labels = [1] * len(mal_files) + [0] * len(ben_files)

    results = []

    for p in tqdm(obfuscation_likelihood, position=0, desc='Running...'):
        results.append(run_experiment(files, labels, model, p, batch_size=512))

    with open('obfuscation-results.json', 'w') as file: 
        json.dump(results, file)