import torch
import datasets
import pandas as pd 
from datasets import DatasetDict 
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers.integrations import WandbCallback
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM

import os 
import pickle
import argparse

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

def get_args():
    parser = argparse.ArgumentParser(description="Configuration for training/evaluating the model.")

    # Model architecture
    parser.add_argument("--max_length", type=int, default=10,
                        help="Max number of tokens in an instruction")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Number of tokens")
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="Size of hidden layers")
    parser.add_argument("--num_hidden", type=int, default=12,
                        help="Number of hidden layers")
    parser.add_argument("--num_attention", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=3072,
                        help="Size of intermediate layers")
    parser.add_argument("--hidden_act", type=str, default="gelu",
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
    parser.add_argument("--split_size", type=float, default=1.0,
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

    return parser.parse_args()

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
    dataset = datasets.load_from_disk("data/tokenized")
    train_subset = dataset['train'].train_test_split(test_size=1 - args.split_size, shuffle=True)['train']
    test_subset = dataset['test'].train_test_split(test_size=args.split_size, shuffle=True)['test']
    dataset = DatasetDict({
        'train': train_subset,
        'test': test_subset,
    })
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
    callback = LogPredictionsCallback(args.eval_data_path, tokenizer)


    trainer = Trainer(
        model=model,
        args=train_args, 
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=dataset['train'], 
        eval_dataset=dataset['test']
    )

    trainer.add_callback(callback)
    trainer.train()
    trainer.save_model(args.model_path)
