from collections import defaultdict 

def handle_sample(sample, **kwargs):
    tokenizer = kwargs.get("tokenizer")
    args = kwargs.get("args")

    if tokenizer is None: 
        raise Exception("Missing tokenizer")
    
    if args is None: 
        raise Exception("Missing CLI args")

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