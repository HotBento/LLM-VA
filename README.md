# LLM-VA: Large Language Model Vector Alignment
This repository contains the code for the paper "LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment".

## Setup
```bash
conda create -n llmva python=3.12.8 -y
conda activate llmva
```


```bash
pip install -r requirements.txt
```

"flash-attn==2.8.2" needs to be installed separately in `https://github.com/Dao-AILab/flash-attention`.


## Usage
Setup server:
```bash
python src/server_answer.py
```

In another terminal, run the client (Use CUDA_VISIBLE_DEVICES to specify which GPUs to use):
```bash
python src/run/llmva_run.py
```
