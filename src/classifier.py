import torch 
import datasets 
import transformers 
import wandb

# from transformers import RobertaForSequenceClassification
# from tokenizers import PreTrainedTokenizerFast 

import string
import random

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

if __name__ == "__main__":
    run = wandb.init()
    artifact = run.use_artifact('henry-williams/malbert-hf/model-malbert-B85b698caa:v2', type='model')
    artifact_dir = artifact.download()

    # model = RobertaForSequenceClassification.from_pretrained("./MalBERTa")
    # tokenizer = PreTrainedTokenizerFast.from_pretrained("./MalBERTa")

    # dataset = datasets.load_from_disk("data/raw")
    # processed_dataset = dataset.map(
    #     handle_sample,
    #     remove_columns=dataset['test'].column_names,
    #     batch_size=64,
    #     batched=True,
    #     num_proc=8,
    # )

    # train_args = TrainingArguments(
    #     output_dir="./MalBERTa-classifier",
    #     overwrite_output_dir=True,
    #     num_train_epochs=1,
    #     per_device_train_batch_size=256, 
    #     logging_steps=100,
    #     report_to="wandb"
    #     run_name=f"malbert-classifier-{''.join([random.choice(string.hexdigits) for _ in range(10)])}"
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=train_args, 
    #     processing_class=tokenizer,
    #     train_dataset=processed_dataset['train'],
    #     eval_dataset=processed_dataset['test'],
    # )

    # trainer.train() 
    # trainer.save_model("./MalBERTa-classifier")