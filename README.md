# Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering


This is the official implementation of our ACL'2022 paper "[Hyperlink-induced Pre-training for Passage Retrieval in OpenQA](https://arxiv.org/pdf/2203.06942.pdf)".

Acknowledgements: our implementation is based on the [DPR](https://github.com/facebookresearch/DPR) repository with minor modifications. Huge thanks to all the authors and contributors of the DPR repository!👏



## Quick Links
- [Overview](https://github.com/jzhoubu/HLP#overview)
- [Setup](https://github.com/jzhoubu/HLP#setup)
  - [Installation](https://github.com/jzhoubu/HLP#installation)
  - [Prepare Data and Models](https://github.com/jzhoubu/HLP#prepare-data-and-models)
- [Experiments](https://github.com/jzhoubu/HLP#experiments) 
  - [Retriever Training](https://github.com/jzhoubu/HLP#retriever-training) 
  - [Corpus Embedding](https://github.com/jzhoubu/HLP#corpus-embedding) 
  - [Retrieval Evalutaion](https://github.com/jzhoubu/HLP#retrieval-evalutaion) 
- [Others](https://github.com/jzhoubu/HLP#others) 
  - [Data Formats for Training Retriever](https://github.com/jzhoubu/HLP#data-formats-for-training-retriever)
  - [Processed Wikipedai Graph](https://github.com/jzhoubu/HLP#our-wikipedia-graph)
- [Citation](https://github.com/jzhoubu/HLP#citation)

## Overview
In this paper, we propose **H**yper**L**ink-induced **P**re-training (HLP), a pre-training method to learn effective Q-P relevance induced by the hyperlink topology within naturally-occurring Web documents. Specifically, these Q-P pairs are automatically extracted from the online documents with relevance adequately designed via hyperlink-based topology to facilitate downstream retrieval for question answering. 

<p align="center"><img width="90%" src="figures/hlp.jpg" /></p>



## Setup
### Installation
1. Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone git@github.com:jzhoubu/HLP.git
cd HLP
conda create -n hlp python=3.7
conda activate hlp
pip install -r requirements.txt
```

2. Please change the `HLP_HOME` variable in `biencoder_train_cfg.yaml`, `gen_embs.yaml` and `dense_retriever.yaml`. The `HLP_HOME` is the path to the HLP directory you download.


3. You may also need to build `apex` via command
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python -m pip install -v --disable-pip-version-check --no-cache-dir ./
```
If you encounter any error in building `apex`, please refer to the official [apex](https://github.com/NVIDIA/apex) repository.


### Prepare Data and Models
**Option1**: Download data via command (Recommand!)
```bash
bash downloader.sh
```
This command will automatically download the necessary data (about 50GB) for experiments. 


**Option2**:  Download data via webpages
After downloading, you need to move them to corresponding subdirectory as below:

**Data Location**
- `${HLP_HOME}/data/train` (train- & val-set)
  - [dl_10m.jsonl](https://drive.google.com/file/d/10YIohcsXAHKFzF2L43qkxH5zYkzTw70R/view?usp=sharing), [cm_10m.jsonl](https://drive.google.com/file/d/10YWz5WN_qJAXVCON47R1cWx2j8MScR1_/view?usp=sharing)
  - [nq-train.jsonl](https://drive.google.com/file/d/1-3fy6UcjVJLt6CW7vRp_OkWb37WMBRBR/view?usp=sharing), [nq-dev.jsonl](https://drive.google.com/file/d/1-4BgqIfd8r-mK8cWP4nunOqowsG0xAJT/view?usp=sharing)
  - [trivia-train.jsonl](https://drive.google.com/file/d/1-5ew6FNHYmauz5YoCKhnAb6wlTNpwEN6/view?usp=sharing), [trivia-dev.jsonl](https://drive.google.com/file/d/1-7qJY872hwoXN9bQQUtbV82BqVCUjSOA/view?usp=sharing)
  - [webq-train.jsonl](https://drive.google.com/file/d/1-7DZ9dPTGIen7_dy4816v4r3fQ-F5h3C/view?usp=sharing),[webq-dev.jsonl](https://drive.google.com/file/d/1-6HgRQ7ocB72rxgsaOhHIkOk176RWfau/view?usp=sharing)

- `${HLP_HOME}/data/eval` (test-set)
  - [nq-test.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv)
  - [trivia-test.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz)
  - [webq-test.qa.csv](https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv)

- `${HLP_HOME}/data/corpus` (retrieval corpus)
  - [psgs_w100.tsv](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz)

If you prefer using other data location, you can change the coerrsponding path in the yaml files under `conf` directory.

**Checkpoint Location**
- `${HLP_HOME}/experiments/hlp20210726/train` 
  - [hlp20210726.best](https://drive.google.com/file/d/10vg8-6S4Rnn7wZOxfBmn2C4il0oV0Oyk/view?usp=sharing)
- `${HLP_HOME}/experiments/hlp20220402/train` (pretrained with latest DPR implementation)
  - [hlp20220402.best](https://drive.google.com/file/d/118DQW2uEK4yUWaTL_4kFzA4K9s-YtHLA/view?usp=sharing) (Recommand✅)

More information of these checkpoints can be found in the [model-card](https://github.com/jzhoubu/HLP/blob/preview/model-card.md).

## Experiments
### Retriever Training
Below is an example to pre-train HLP. 

```bash
python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
    hydra.run.dir=./experiments/pretrain_hlp/train \
    val_av_rank_start_epoch=0 \
    train_datasets=[dl,cm] dev_datasets=[nq_dev] \
    train=pretrain_8xV100
```
- `hydra.run.dir`: working directory of hydra (logs and checkpoints will be saved here). 
- `val_av_rank_start_epoch`: epoch number when we start use average ranking for validation. 
- `train_datasets`: alias of the train set name (see `conf/datasets/train.yaml`). 
- `dev_datasets`: alias of the dev set name (see `conf/datasets/train.yaml`). 
- `train`: a yaml file of training configuration (under `conf/train`)
- See more configuration setting in `biencoder_train_cfg.yaml` and `pretrain_8xV100.yaml`. 


Below is an example to fine-tune on NQ dataset using a pre-trained checkpoint:
```bash
python -m torch.distributed.launch --nproc_per_node=8 train_dense_encoder.py \
    hydra.run.dir=./experiments/finetune_nq/train \
    model_file=../../pretrain_hlp/train/dpr_biencoder.best \
    train_datasets=[nq_train] dev_datasets=[nq_dev] \
    train=pretrain_8xV100
```
- `model_file`: a relative path to the model checkpoint



### Corpus Embedding
Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.

Below is an example to generate embeddings of the wikipedia corpus.
```bash
python ./generate_dense_embeddings.py \
    hydra.run.dir=./experiments/pretrain_hlp/embed \
    train=pretrain_8xV100 \
    model_file=../train/dpr_biencoder.best \
    ctx_src=dpr_wiki \
    shard_id=0 num_shards=1 \
    out_file=embedding_dpr_wiki \
    batch_size=10000
```
- `model_file`: a relative path to the model checkpoint.
- `ctx_src`: alias of the passages resource (see `conf/ctx_sources/corpus.yaml`).
- `out_file`: prefix name of the output embedding.
- `shard_id`: number(0-based) of data shard to process
- `num_shards`: total amount of data shards


### Retrieval Evalutaion
Below is an example to evaluate a model on NQ test set.
```bash
python dense_retriever.py \
	  hydra.run.dir=./experiments/pretrain_hlp/infer \
	  train=pretrain_8xV100 \
	  model_file=../train/dpr_biencoder.best \
	  qa_dataset=nq_test \
	  ctx_datatsets=[dpr_wiki] \
	  encoded_ctx_files=["../embed/embedding_dpr_wiki*"] \
	  out_file=nq_test.result \
```
- `model_file`: a relative path to the model checkpoint
- `qa_dataset`: alias of the test set (see `conf/datasets/eval.yaml`)
- `encoded_ctx_files`: list of corpus embedding files glob expression
- `out_file`: path of the output file



## Others

### Data Formats for Training Retriever
Below shows data format of our train and dev data (i.e. `dl_10m.jsonl` and `nq-train.json`). Our implementation can work with json and jsonl files. 
More format descriptions can refer to [here](https://github.com/facebookresearch/DPR#resources--data-formats).

```
[
  {
	"question": "....",
	"positive_ctxs": [{"title": "...", "text": "...."}],
	"negative_ctxs": [{"title": "...", "text": "...."}],
	"hard_negative_ctxs": [{"title": "...", "text": "...."}]
  },
  ...
]
```

### Processed Wikipedai Graph
We also release our processed wikipedia graph which considers passages as nodes and hyperlinks as links. Further details can be found in the Section 3 in our paper. Click [here](https://drive.google.com/file/d/1-1v3_rsby0lQnduOw1YRIvrRVBV2xbnP/view?usp=sharing) to download. 

```python
import json, glob
from tqdm import tqdm
PATH = "/home/data/jzhoubu/wiki_20210301_processed/**/wiki_**.json" # change this path accordingly
files = glob.glob(PATH)
title2info = {}
for f in tqdm(files):
    sample = json.load(open(f, "r"))
    for k,v in sample.items():
        title2info[k] = v

print(len(title2info.keys())) 
# 22334994

print(title2info['Anarchism_0'])
# {'text': 
#    'Anarchism is a <SOE> political philosophy <EOE> and <SOE> movement <EOE> that is sceptical of <SOE> authority <EOE> and rejects all involuntary, coercive forms of <SOE> hierarchy <EOE> . Anarchism calls for the abolition of the <SOE> state <EOE> , which it holds to be undesirable, unnecessary, and harmful. It is usually described alongside <SOE> libertarian Marxism <EOE> as the libertarian wing ( <SOE> libertarian socialism <EOE> ) of the socialist movement and as having a historical association with <SOE> anti-capitalism <EOE> and <SOE> socialism <EOE> . The <SOE> history of anarchism <EOE> goes back to <SOE> prehistory <EOE> ,',
# 'mentions': 
#    ['political philosophy', 'movement', 'authority', 'hierarchy', 'state', 'libertarian Marxism', 'libertarian socialism', 'anti-capitalism', 'socialism', 'history of anarchism', 'prehistory'],
# 'linkouts': 
#    ['Political philosophy', 'Political movement', 'Authority', 'Hierarchy', 'State (polity)', 'Libertarian Marxism', 'Libertarian socialism', 'Anti-capitalism', 'Socialism', 'History of anarchism', 'Prehistory']
# }

```



## Citation

If you find this work useful, please cite the following paper:

```
@article{zhou2022hyperlink,
  title={Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering},
  author={Zhou, Jiawei and Li, Xiaoguang and Shang, Lifeng and Luo, Lan and Zhan, Ke and Hu, Enrui and Zhang, Xinyu and Jiang, Hao and Cao, Zhao and Yu, Fan and others},
  journal={arXiv preprint arXiv:2203.06942},
  year={2022}
}
```