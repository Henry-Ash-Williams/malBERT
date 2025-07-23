import json
import torch
import wandb
import r2pipe
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, PreTrainedTokenizerFast

import os
import random
import tempfile
import subprocess
from os import PathLike
from argparse import ArgumentParser
from typing import Dict, List, Tuple

device = torch.device("mps")

RUNS = {"frozen": "47y92682", "pretrained": "daf0h543", "base": "wozkyaa6"}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"


def get_args():
    parser = ArgumentParser(
        prog="Obfuscation Experiment",
        description="TODO: Write a description",
    )

    parser.add_argument(
        "target",
        choices=["malicious", "benign", "all"],
        default="all",
        type=str,
        help="The class of executables to be obfuscated",
    )

    parser.add_argument(
        "-m",
        "--model",
        choices=["frozen", "pretrained", "base"],
        default="base",
        type=str,
        help="The model used by the experiment",
    )

    parser.add_argument("-b", "--batch-size", type=int, default=512, help="Batch size")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Results output location",
    )

    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=10,
        help="Number of experiment steps",
    )

    parser.add_argument(
        "-f",
        "--files",
        type=str,
        help="Path to a JSON file containing the locations and labels of files used by the experiment",
        default="/Users/henrywilliams/Documents/programming/python/ai/malbert-test/notebooks/obfuscation-experiment-files.json",
    )

    parser.add_argument(
        "-t",
        "--tokenizer-path",
        type=str,
        help="Path to the tokenizer",
        default="/Users/henrywilliams/Documents/programming/python/ai/malbert-test/MalBERTa",
    )

    args = parser.parse_args()
    return args


def get_model(id: str) -> RobertaForSequenceClassification:
    api = wandb.Api()

    with tempfile.TemporaryDirectory() as tdir:
        artifact = api.artifact(
            f"henry-williams/opcode-malberta/model-{id}:v1", type="model"
        )
        path = artifact.download(root=tdir)

        return RobertaForSequenceClassification.from_pretrained(path)


def get_disassm(path: PathLike) -> List[Dict[str, str]]:
    r2 = r2pipe.open(path, ["-12"])
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


def make_opcode_sequences(instrs: List[Dict[str, str]]) -> List[str]:
    valid_instrs = [
        instruction["inst"]
        for instruction in instrs
        if instruction["inst"] != "invalid"
    ]

    return " ".join([instr.split(" ")[0] for instr in valid_instrs])


def tokenize(
    sample: str, model: RobertaForSequenceClassification
) -> Dict[str, List[int]]:
    seq_length = model.config.max_position_embeddings

    input = tokenizer(
        sample,
        padding="max_length",
        max_length=seq_length - 2,
        return_overflowing_tokens=True,
        truncation=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )

    return input


def pipeline(
    args, model: RobertaForSequenceClassification, path: os.PathLike
) -> Tuple[torch.Tensor, torch.Tensor]:
    # if not os.path.exists(path):
    #     raise Exception(f"Could not find specified file at {path}")

    disassembly = get_disassm(path)
    opcodes = make_opcode_sequences(disassembly)
    input = tokenize(opcodes, model)

    model.eval()

    logits = []
    input_ids = input["input_ids"].split(args.batch_size)
    attention_mask = input["attention_mask"].split(args.batch_size)
    token_type_ids = input["token_type_ids"].split(args.batch_size)

    torch.mps.empty_cache()

    for ids, attn_mask, tok_ty_ids in tqdm(
        zip(input_ids, attention_mask, token_type_ids),
        total=len(input_ids),
        position=2,
        leave=False,
        desc=path.split("/")[-1],
    ):
        ids = ids.to(device)
        attn_mask = attn_mask.to(device)
        tok_ty_ids = tok_ty_ids.to(device)

        with torch.no_grad():
            logits.append(
                model(
                    input_ids=ids, attention_mask=attn_mask, token_type_ids=tok_ty_ids
                )
            )

    logits = torch.vstack([logit.logits for logit in logits])
    return logits


def obfuscate(file: os.PathLike, output: os.PathLike) -> os.PathLike:
    obfuscated_path = os.path.join(
        output, file.split("/")[-1].replace(".exe", ".obfs.exe")
    )
    subprocess.run(
        ["upx", "-o", obfuscated_path, file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return obfuscated_path


def experiment_step(
    args,
    model: RobertaForSequenceClassification,
    file: os.PathLike,
    label: int,
    tdir: str,
    p: float = 1.0,
) -> Tuple[torch.Tensor, bool]:
    """
    p is the likelihood of the file being obfuscated
    """
    obfuscated = False
    q = random.random()

    if (
        args.target == "all"
        or (args.target == "malicious" and label == 1)
        or (args.target == "benign" and label == 0)
    ):
        if q < p:
            path = obfuscate(file, tdir)
            obfuscated = True
        else:
            path = file
    else:
        path = file

    return pipeline(args, model, path), obfuscated


def run_experiment(
    args,
    files: List[os.PathLike],
    labels: List[int],
    model: RobertaForSequenceClassification,
    step: int,
    p: float = 0.0,
) -> Dict[str, Dict[str, float] | float]:
    predicted = []
    actual = []

    os.makedirs(os.path.join(args.output, f"step-{step}/logits"), exist_ok=True)

    with tempfile.TemporaryDirectory() as tdir:
        for file, label in tqdm(
            zip(files, labels),
            total=len(files),
            leave=False,
            desc=f"p = {p:.2}",
            position=1,
        ):
            try:
                logits, was_obfuscated = experiment_step(
                    args, model, file, label, tdir, p=p
                )
            except Exception as e:
                # print(e)
                continue

            output_logits_path = os.path.join(
                args.output,
                f"step-{i}/logits",
                f"{file.split('/')[-1].replace('.exe', '.obfs.pt' if was_obfuscated else '.pt')}",
            )

            torch.save(logits, output_logits_path)
            class_likelihood = F.softmax(logits.mean(dim=0), dim=0)

            predicted.append(class_likelihood.argmax().item())
            actual.append(label)

        torch.save(
            torch.tensor(predicted),
            os.path.join(args.output, f"step-{step}/predicted.pt"),
        )
        torch.save(
            torch.tensor(actual), os.path.join(args.output, f"step-{step}/actual.pt")
        )

    return classification_report(actual, predicted, output_dict=True)


if __name__ == "__main__":
    args = get_args()
    model = get_model(RUNS[args.model]).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(args.__dict__, f)

    with open(args.files, "r") as file:
        data = json.load(file)

    files = list(data.keys())
    labels = list(data.values())

    obfuscation_likelihood = np.linspace(0.0, 1.0, args.steps)

    results = []

    for i, p in tqdm(
        enumerate(obfuscation_likelihood),
        position=0,
        desc="Running...",
        total=len(obfuscation_likelihood),
    ):
        results.append(run_experiment(args, files, labels, model, i, p))

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f)
