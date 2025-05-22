import wandb
import torch
import datasets
import pandas as pd 
from datasets import DatasetDict 
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

import os 
import pickle
import argparse
import datetime
from collections import defaultdict

# Set up weights & biases 
os.environ["WANDB_PROJECT"] = "malbert-hf"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# Weights & Biases callback to log evaluation samples
class LogPredictionsCallback(WandbCallback):
    def __init__(self, data_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, 'rb') as data_file: 
            self.data = pickle.load(data_file)
        
    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        device = model.device 
        
        model.eval()
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor(self.data['input_ids']).to(device),
                attention_mask=torch.tensor(self.data['attention_mask']).to(device),
                labels=torch.tensor(self.data['labels']).to(device)
            )

        self.data['input_ids'] = torch.tensor(self.data['input_ids']).detach()
        self.data['attention_mask'] = torch.tensor(self.data['attention_mask']).detach()
        self.data['labels'] = torch.tensor(self.data['labels']).detach()

        mask_pos = torch.where(self.data['input_ids'] == self.tokenizer.mask_token_id)

        input = torch.clone(self.data['input_ids'])
        actual = torch.clone(self.data['input_ids'])
        predicted = torch.clone(self.data['input_ids'])

        actual[actual == self.tokenizer.mask_token_id] = self.data['labels'][mask_pos[0], mask_pos[1]]
        predicted[predicted == self.tokenizer.mask_token_id] = output.logits[mask_pos[0], mask_pos[1], :].argmax(dim=-1).cpu()

        x = [self.tokenizer.decode(xi[~torch.isin(xi, torch.tensor([0, 1, 2, 3]))]) for xi in input]
        y = self.tokenizer.batch_decode(actual, skip_special_tokens=True)
        y_hat = self.tokenizer.batch_decode(predicted, skip_special_tokens=True)

        df = pd.DataFrame({"Input": x, "Actual": y, "Predicted": y_hat})
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
    parser.add_argument("--vocab_size", type=int, default=10000,
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
    parser.add_argument("--eval_data_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/data.pickle",
                        help="Evaluation subset")
    parser.add_argument("--dataset_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/data/tokenized",
                        help="Tokenized dataset path")
    parser.add_argument("--model_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/MalBERT",
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
        prediction_loss_only=True,  
        report_to="wandb",
        run_name="malbert",
        eval_strategy="steps",
        eval_steps=10000,
    )

    print("Configuration:")
    [print(f"  {k:<30}  {v}") for k, v in args_dict.items()]

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

    print("Loading tokenizer...", end="")
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_path)
    print(" done")

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


    trainer = Trainer(
        model=model,
        args=train_args, 
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset['train'], 
        eval_dataset=dataset['test'],
        callbacks=[
            LogPredictionsCallback(args.eval_data_path, tokenizer),
            AbortIfTooSlow((len(dataset['train']) // args.batch_size) * args.epochs, min_fraction=1.0, max_time_hours=0.007)
        ]
    )

    trainer.train()
    trainer.save_model(args.model_path)
