# MalBERTa

RoBERTa for static malware detection. Part of my masters thesis at the University of Sussex. 

Pre-trains RoBERTa using MaskedLM on x86 opcode sequences, then trains a sequence classifier

## Setup 

1. Clone the repo 

```sh
$ git clone git@github.com:Henry-Ash-Williams/MalBERT
$ cd MalBERT
```

2. Install `uv` 

```sh
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies 

```sh
$ uv sync
```

## File Structure 

```
.
├── README.md
├── MalBERTa            # Tokenizer 
├── notebooks           # Jupyter notebooks for data analysis and testing 
├── src                 # Python scripts 
│   ├── pretrain.py     # Pretrainer
│   └── classifier.py   # Classifier 
├── data                # Dataset 
│   └── raw.zip       
├── pyproject.toml      # Project spec
├── malberta-runs.csv   # Run history 
└── uv.lock             # UV metadata
```


