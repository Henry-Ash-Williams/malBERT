import wandb
import torch
import datasets
import pandas as pd 
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import RobertaConfig, PreTrainedTokenizerFast, RobertaForMaskedLM

import os 
import argparse
import datetime
from collections import defaultdict

# Set up weights & biases 
os.environ["WANDB_PROJECT"] = "malbert-hf"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Weights & Biases callback to log evaluation samples
class LogPredictionsCallback(WandbCallback):
    def __init__(self, data_collator, token_ids, tokenizer):
        super().__init__()
        self.data_collator = data_collator 
        self.token_ids = token_ids 
        self.tokenizer = tokenizer
        
    def on_train_end(self, args, state, control, **kwargs):
        X = self.data_collator(torch.tensor(self.token_ids['input_ids']))
        preds = trainer.predict(X['input_ids'])
        
        Y_hat = tokenizer.batch_decode(preds.predictions.argmax(-1))
        Y = tokenizer.batch_decode(self.token_ids['input_ids'])

        df = pd.DataFrame(data={
            "Input": tokenizer.batch_decode(X['input_ids']),
            "Predicted": Y_hat,
            "Actual": Y,
        })

        table = self._wandb.Table(dataframe=df)
        self._wandb.log({"sample": table})

class AbortIfTooSlow(TrainerCallback):
    def __init__(self, total_steps: int, min_fraction: float = 0.1, max_time_hours: int = 1):
        self.total_steps = total_steps 
        self.min_steps = int(total_steps * min_fraction)
        self.max_time = datetime.timedelta(hours=max_time_hours)
        self.start_time = None 
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = datetime.datetime.now() 
        
    def on_step_end(self, args, state, control, **kwargs):
        elapsed = datetime.datetime.now() - self.start_time 
        
        if elapsed > self.max_time and state.global_step < self.min_steps: 
            print(f"‼️ Training too slow: only {state.global_step} out of {self.min_steps}", 
                  f"after {elapsed} seconds. Aborting... ")

            if wandb.run is not None:
                wandb.run.summary["status"] = "FAILED"
                wandb.run.summary["failure_reason"] = "Training was too slow, aborted"
                wandb.run.finish(exit_code=1)
            
            control.should_training_stop = True

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

    # Not really necessary tbh 
    parser.add_argument("--max_length", type=int, default=10,
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
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Training batch size")
    parser.add_argument("--split_size", type=float, default=1.0 - 1e-5,
                        help="Portion of training and testing data to use")

    # Paths
    parser.add_argument("--dataset_path", type=str,
                        default="/Users/henrywilliams/Documents/programming/python/ai/malbert-test/data",
                        help="Tokenized dataset path")
    parser.add_argument("--model_path", type=str,
                        default="/Users/henrywilliams/Documents/programming/python/ai/malbert-test/MalBERTa",
                        help="Path to tokenizer")

    args = parser.parse_args()
    args.hidden_size = args.num_attention * args.hidden_size_factor 
    del args.hidden_size_factor 

    args.intermediate_size = args.hidden_size * args.intermediate_size_factor 
    del args.intermediate_size_factor 
    
    return args 

if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)

    train_args = TrainingArguments(
        output_dir=args.model_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        save_strategy="epoch",
        # prediction_loss_only=True,  
        report_to="wandb",
        run_name="malbert",
        eval_strategy="steps",
        eval_steps=10000,
    )

    print("Configuration:")
    [print(f"  {k:<30}  {v}") for k, v in args_dict.items()]

    print("Loading tokenizer...", end="")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_path)
    print(" done")

    print("Loading data...", end="")
    dataset = datasets.load_from_disk("data/raw")
    processed_dataset = dataset.map(
        handle_sample,
        remove_columns=dataset['test'].column_names,
        batch_size=64,
        batched=True,
        num_proc=8,
    )
    print(" done.")
    print(f"Dataset has {len(dataset['train'])} samples in train, {len(dataset['test'])} in test")


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


    train_ds = processed_dataset['train'].remove_columns('label')
    test_ds = processed_dataset['test'].remove_columns('label')
    pred_data = test_ds.select(range(10))

    trainer = Trainer(
        model=model,
        args=train_args, 
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds, 
        eval_dataset=test_ds,
        callbacks=[
            LogPredictionsCallback(data_collator, pred_data, tokenizer),
            AbortIfTooSlow((len(dataset['train']) // args.batch_size) * args.epochs, min_fraction=1.0, max_time_hours=0.007)
        ]
    )

    trainer.train()
    trainer.save_model(args.model_path)
