# **Model Card: HLP**
Inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993) and [Lessons from Archives (Jo & Gebru)](https://arxiv.org/pdf/1912.10389.pdf), we’re providing some accompanying information about the pre-trained HLP retriever.


### **Model Details**
Our HLP model was pre-trained on 20m DL and CM pseudo Q-P pairs extracted from Wikipedia graph. The details involved in training the HLP are described in the paper. We release two HLP checkpoints in our repository:
- The original version is trained on July 2021, using a old (at Dec 2020) DPR framework. 
- The lastest version is trained on April 2022, using the lastest DPR framework. Our repository is built on the lastest framework.

They acchieve closed performance which is shown below.

|  | NQ | TriviaQA | WQ    |
|:-----:|:----: |:----: |:----: |
| Zero-shot<br>Fine-tune | top1 / top5 / top100 | top1 / top5 / top100  | top1 / top5 / top100  |
| **HLP (origin)** | 51.2 / 70.2 / 82.0<br>70.9 / 81.4 / 88.0 | 65.9 / 76.9 / 84.0<br>75.3 / 82.4 / 86.9 | 49.3 / 66.9 / 80.8<br>65.5 / 76.5 / 84.5 |
| **HLP (latest)** | 50.9 / 69.3 / 82.1<br>70.6 / 81.3 / 88.0 | / |  / | 
||||




### **Model Date**
Origian version: July 2021 <br>
Latest  version: April 2022 (Recommand!✅)


### **Training Data** (Pre-training)
We choose the snapshot 03-01-2021 of an English Wikipedia dump, and process it with [WikiExtractor](https://github.com/attardi/wikiextractor), resulting in over 22 million passages. We then extract 20 million pseudo Q-P pairs (10m dual-link and 10m co-mention) which our model is pre-trained on. More detail can be found in the Section 3 of our paper. We also release the training data and the processed Wikipedia graph in our homepage. Feel free to have a try.



### **Evaluation Data** (Downstream)

- **NQ/Trivia/WebQ** 
  - Source of [train/dev/test/corpus]: the [DPR](https://github.com/facebookresearch/DPR) repository. <br>
  - Note: For each sample, we only keep the top1 positive ctx and remove all negatvies. 


- **HotpotQA** (full wiki)
  - Source of [train/dev/corpus]: [HotpotQA Homepage](https://hotpotqa.github.io/).  <br>
  - Note: We use the negatives provided by [Path Retriever](https://github.com/AkariAsai/learning_to_retrieve_reasoning_paths) (Akari Asai et al. 2020).


- **MS MARCO** (document ranking)
  - Source of [dev/corpus]:  [MSMARCO](https://github.com/microsoft/MSMARCO-Document-Ranking) repository.



- **BioASQ** (factoid)<br>
  - Source of [dev]: [BioASQ](http://participants-area.bioasq.org/datasets/) dataset. <br>
  - Source of [corpus]: [AugDPR](https://arxiv.org/abs/2104.07800) (Revanth Gangi Reddy et al. 2021) (i.e. We use the same passage split to AugDPR).
