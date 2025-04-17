# MalBERT

RoBERTa for static malware detection. Part of my masters thesis at the University of Sussex. 

Pre-trains RoBERTa using MaskedLM on disassembled x86 instructions, then fine-tunes on a downstream classification problem. 

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
├── data                # Dataset 
│   └── README.md       
├── data.pickle         # Evaluation data
├── pretrain.ipynb      # Pre-training
├── pyproject.toml      # Project spec
└── uv.lock             # UV metadata
```


