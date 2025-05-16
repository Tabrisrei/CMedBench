# CMedBench: Comprehensive Evaluation of the LLM Compression Impact on Medical Applications

<!-- ![image-20241026195404186](./figs/f1.png) -->

## Introduction

<!-- [LLMCBench: Benchmarking Large Language Model Compression for Efficient Deployment [arXiv]](https://arxiv.org/abs/2410.21352) -->

 The **C**compressed **Med**ical LLM **Bench**mark (CMedBench) is a rigorously designed benchmark with an in-depth analysis for evaluating compressed LLMs in medical context. 

## Installation

```
git clone https://github.com/Tabrisrei/CMedBench.git
cd CMedBench

conda create -n cmedbench python=3.10
conda activate cmedbench
cd TrustLLM/trustllm_pkg
pip install -e .
cd ../../opencompass
pip install -e .
```


## Get Dataset
### Track-1/2/4
Add your token to PycrawlersDownload.py

```
python PycrawlersDownload.py
```
We also provide dataset zip file, you can unzip it in the folder to get all the track1/2 datasets.

### Track-3
```
unzip TrustLLM/dataset/dataset.zip to the folder to get all the Trustworthy datasets
```

## Usage

This repo contains scripts for testing all the five Tracks

#### Testing MMLU

```
bash scripts/run_mmlu.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--data_dir` : Dataset location.
- `--ntrain` : number of shots.
- `--seqlen` : Denotes the maximum input sequence length for LLM.

#### Testing MNLI

```
bash scripts/run_mnli.sh
```

##### Overview of Arguments:

- `--path` : Model checkpoint location.
- `--seqlen` : Denotes the input sequence length for the model.

