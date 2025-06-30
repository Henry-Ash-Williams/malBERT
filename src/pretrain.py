import datasets
import wandb
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaConfig, PreTrainedTokenizerFast, RobertaForMaskedLM

from utils import handle_sample
import os 
import random
import argparse
from string import hexdigits 

# Set up weights & biases 
os.environ["WANDB_PROJECT"] = "opcode-malberta"
os.environ["WANDB_LOG_MODEL"] = "end"

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

    # Paths
    parser.add_argument("--dataset_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/data/raw",
                        help="Tokenized dataset path")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/its/home/hw452/programming/MalBERT/MalBERTa",
                        help="Path to tokenizer")
    parser.add_argument("--output_path", type=str, 
                        help="Where to save the model",
                        required=True)
    parser.add_argument("--dataset_size", type=float, 
                        help="Portion of the dataset to use",
                        default=1.0)

    args = parser.parse_args()
    args.hidden_size = args.num_attention * args.hidden_size_factor 
    del args.hidden_size_factor 

    args.intermediate_size = args.hidden_size * args.intermediate_size_factor 
    del args.intermediate_size_factor 
    
    return args 

def do_pretrain(args):
    wandb.init(
        project=os.environ["WANDB_PROJECT"],
        name=f"malberta-{''.join([random.choice(hexdigits) for _ in range(10)])}",
        tags=["pretrain"], 
    )
    train_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, 
        per_device_eval_batch_size=args.batch_size,
        logging_steps=100,
        save_strategy="epoch",
        report_to="wandb",
        eval_strategy="epoch",
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.tokenizer_path)

    print("Loading data...", end="")
    dataset = datasets.load_from_disk(args.dataset_path)

    if args.dataset_size < 1.0: 
        dataset["test"] = dataset["test"].shuffle().select(range(int(len(dataset["test"]) * args.dataset_size)))
        dataset["train"] = dataset["train"].shuffle().select(range(int(len(dataset["train"]) * args.dataset_size)))

    processed_dataset = dataset.map(
        handle_sample,
        remove_columns=dataset['test'].column_names,
        batch_size=64,
        batched=True,
        num_proc=8,
        fn_kwargs=dict(tokenizer=tokenizer, args=args)
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

    trainer = Trainer(
        model=model,
        args=train_args, 
        processing_class=tokenizer,
        data_collator=data_collator,
        train_dataset=train_ds, 
        eval_dataset=test_ds,
    )

    trainer.train()
    trainer.save_model(args.output_path)

if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)

    print("Configuration:")
    [print(f"  {k:<30}  {v}") for k, v in args_dict.items()]

    do_pretrain(args)
