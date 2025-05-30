import wandb
import torch
import datasets
import pandas as pd 
from datasets import disable_caching
from transformers.integrations import WandbCallback
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaForMaskedLM
from transformers import PreTrainedTokenizerFast

import os 
import random
import argparse
import datetime
from string import hexdigits 
from collections import defaultdict


# Set up weights & biases 
os.environ["WANDB_PROJECT"] = "opcode-malberta"
os.environ["WANDB_LOG_MODEL"] = "end"
# os.environ["WANDB_WATCH"] = "all"

pretrain_eval_loss = None

class AbortIfTooSlow(TrainerCallback):
    def __init__(self, total_steps: int, min_fraction: float = 0.1, max_time_hours: int = 1):
        self.total_steps = total_steps 
        self.min_steps = int(total_steps * min_fraction)
        self.max_time = datetime.timedelta(hours=max_time_hours)
        self.start_time = None 
        self._train_time = datetime.timedelta()
        self._last_step_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self._train_time = datetime.timedelta() 
        self._last_step_time = datetime.datetime.now() 

    def on_step_begin(self, args, state, control, **kwargs):
        self._last_step_time = datetime.datetime.now() 
        
    def on_step_end(self, args, state, control, **kwargs):
        now = datetime.datetime.now() 

        if self._last_step_time is not None: 
            self._train_time += now - self._last_step_time 
            
        if self._train_time > self.max_time and state.global_step < self.min_steps: 
            print(f"‼️ Training too slow: only {state.global_step} out of {self.min_steps}", 
                  f"after {self._train_time} seconds. Aborting... ")

            raise RuntimeError("Training took too long")


def handle_sample(sample):
    texts = sample['text']
    labels = sample['label']
    
    flattened = defaultdict(list)

    for text, label in zip(texts, labels):
        tokenized = tokenizer(
            text,
            padding='max_length',
            max_length=args.max_length,
            return_overflowing_tokens=True,
            truncation=True
        )

        for i in range(len(tokenized['input_ids'])):
            for k in tokenized:
                flattened[k].append(tokenized[k][i])
            flattened['label'].append(label)

    return dict(flattened)
        

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for training/evaluating the model.")

    parser.add_argument("--max_length", type=int, default=64,
                        help="Max number of tokens in an instruction")
    parser.add_argument("--vocab_size", type=int, default=1293,
                        help="Number of tokens")
    
    # Model architecture
    parser.add_argument("--hidden_size_factor", type=int, default=64,
                        help="Multiplier for number of attention heads to get hidden size")
    parser.add_argument("--num_hidden", type=int, default=12,
                        help="Number of hidden layers")
    parser.add_argument("--num_attention", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--intermediate_size_factor", type=int, default=4,
                        help="Multiplier for size of hidden layers to get intermediate ")
    parser.add_argument("--hidden_act", type=str, default="gelu", choices=["gelu", "relu", "silu", "gelu_new"],
                        help="Activation function used in the hidden layers")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1,
                        help="Dropout probability in hidden layers")
    parser.add_argument("--attention_dropout_prob", type=float, default=0.1,
                        help="Dropout probability in attention mechanisms")
    parser.add_argument("--init_range", type=float, default=0.02,
                        help="Variance of initialisation")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-12,
                        help="Epsilon value used by layernorm")

    # Training
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--do_pretrain", action="store_true", default=False)
    parser.add_argument("--notes", default=None, type=str)

    # Paths
    parser.add_argument("--dataset_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/data/raw",
                        help="Tokenized dataset path")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/MalBERTa",
                        help="Path to tokenizer")
    parser.add_argument("--output_path", type=str, 
                        help="Where to save the model")

    args = parser.parse_args()
    args.hidden_size = args.num_attention * args.hidden_size_factor 
    del args.hidden_size_factor 

    args.intermediate_size = args.hidden_size * args.intermediate_size_factor 
    del args.intermediate_size_factor 
    
    return args 

def pretrain(dataset, tokenizer, args):
    print("Loading model...", end="")
    config = RobertaConfig(
        vocab_size=args.vocab_size, 
        max_position_embeddings=args.max_length + 2, 
        num_attention_heads=args.num_attention,
        num_hidden_layers=args.num_hidden,
        type_vocab_size=1,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        hidden_act=args.hidden_act,
        hidden_dropout_prob=args.hidden_dropout_prob, 
        attention_probs_dropout_prob=args.attention_dropout_prob,
        initializer_range=args.init_range, 
        layer_norm_eps=args.layer_norm_eps,
    )
    model = RobertaForMaskedLM(config=config)    
    print(" done")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    train_ds = dataset['train'].remove_columns('label')
    test_ds = dataset['test'].remove_columns('label')

    train_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size,
        save_total_limit=1,
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=train_args, 
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds, 
        eval_dataset=test_ds,
        callbacks=[
            AbortIfTooSlow((len(dataset['train']) // args.batch_size) * args.epochs, min_fraction=0.1, max_time_hours=1)
        ]
    )

    trainer.train()
    trainer.save_model(args.output_path)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }

if __name__ == "__main__":
    disable_caching()
    args = get_args()
    args_dict = vars(args)


    print("Configuration:")
    [print(f"  {k:<30}  {v}") for k, v in args_dict.items()]

    print("Loading tokenizer...", end="")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)
    print(" done")

    print("Loading data...", end="")
    dataset = datasets.load_from_disk(args.dataset_path)

    dataset["test"] = dataset["test"].shuffle().select(range(int(len(dataset["test"]) * 0.1)))
    dataset["train"] = dataset["train"].shuffle().select(range(int(len(dataset["train"]) * 0.1)))

    processed_dataset = dataset.map(
        handle_sample,
        remove_columns=dataset['test'].column_names,
        batch_size=64,
        batched=True,
        num_proc=8,
    )
    print(" done.")
    print(f"Dataset has {len(processed_dataset['train'])} samples in train, {len(processed_dataset['test'])} in test")

    if args.do_pretrain:
        pretrain(processed_dataset, tokenizer, args) 
        classifier_model = RobertaForSequenceClassification.from_pretrained(args.output_path)
    else: 
        config = RobertaConfig(
            vocab_size=args.vocab_size, 
            max_position_embeddings=args.max_length + 2, 
            num_attention_heads=args.num_attention,
            num_hidden_layers=args.num_hidden,
            type_vocab_size=1,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            hidden_act=args.hidden_act,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_dropout_prob,
            initializer_range=args.init_range, 
            layer_norm_eps=args.layer_norm_eps,
        )
        classifier_model = RobertaForSequenceClassification(config)


    wandb.init(
        project="opcode-malberta",
        name=f"malbert-classifier-{''.join([random.choice(hexdigits) for _ in range(10)])}",
        tags=['pretrained'] if args.do_pretrain else None,
        notes=args.notes,
    )
    train_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        save_total_limit=1,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size * 2, 
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=100,
        report_to="wandb",
    )

    trainer = Trainer(
        model=classifier_model,
        args=train_args, 
        processing_class=tokenizer,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train() 
    trainer.save_model(args.output_path)
        