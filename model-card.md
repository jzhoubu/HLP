# **Model Card: HLP**

### **Model Details**
Our HLP model was pre-trained on 20m DL and CM pseudo Q-P pairs extracted from Wikipedia graph. The details involved in training the HLP are described in the paper. 
In this repository, the released checkpoint is reproduced on April 2022, using the latest transformers and DPR framework, which results in a slight difference.



<table width="200%" align="left" border="0" cellspacing="0" cellpadding="0" frame=void rules=none>
<tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th width="10%" rowspan="3" valign="center">Models</th>
<th width="10%" rowspan="3" valign="center">Trainset</th>
<th width="10%" rowspan="3" valign="center">TrainConfig</th>
<th width="20%" rowspan="3" valign="center">Size</th>
<th width="60%" colspan="9" valign="center">Zero-shot Performance</th>

<tr>
  <td width="20%" colspan="3" align="center"> NQ </td>
  <td width="20%" colspan="3" align="center"> TriviaQA </td>
  <td width="20%" colspan="3" align="center"> WebQ </td>
</tr>

<tr>
  <td width="20%"  align="center"> Top5 </td>
  <td width="20%"  align="center"> Top20 </td>
  <td width="20%"  align="center"> Top100 </td>
  <td width="20%"  align="center"> Top5 </td>
  <td width="20%"  align="center"> Top20 </td>
  <td width="20%"  align="center"> Top100 </td>
  <td width="20%"  align="center"> Top5 </td>
  <td width="20%"  align="center"> Top20 </td>
  <td width="20%"  align="center"> Top100 </td>
</tr>

<tr>
  <td width="10%" align="center"> BM25 </td>
  <td width="10%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> 43.6 </td>
  <td width="20%" align="center"> 62.9 </td>
  <td width="20%" align="center"> 78.1 </td>
  <td width="20%" align="center"> 66.4 </td>
  <td width="20%" align="center"> 76.4 </td>
  <td width="20%" align="center"> 83.2 </td>
  <td width="20%" align="center"> 42.6 </td>
  <td width="20%" align="center"> 62.8 </td>
  <td width="20%" align="center"> 76.8 </td>
</tr>



<tr>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/d/1-2VCWwZepRPLjs0l-nhT40mSOLEfBSgc/view?usp=sharing"> DL </a> 
  </td>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/d/10YIohcsXAHKFzF2L43qkxH5zYkzTw70R/view?usp=sharing"> dl_10m </a>
  </td>
  <td width="20%" align="center"> 
  <a href="https://github.com/jzhoubu/HLP/blob/master/conf/train/pretrain_8xV100.yaml"> pretrain_8xV100 </a> 
  </td>
  <td width="20%" align="center"> 418M </td>
  <td width="20%" align="center"> 49.0 </td>
  <td width="20%" align="center"> 67.8 </td>
  <td width="20%" align="center"> 79.7 </td>
  <td width="20%" align="center"> 62.0 </td>
  <td width="20%" align="center"> 73.8 </td>
  <td width="20%" align="center"> 82.1 </td>
  <td width="20%" align="center"> 48.4 </td>
  <td width="20%" align="center"> 67.1 </td>
  <td width="20%" align="center"> 79.5 </td>
</tr>

<tr>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/d/1-10eOZ0W86kkz3X33_dsrjtxy-Rht6ts/view?usp=sharing"> CM </a> 
  </td>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/d/10YWz5WN_qJAXVCON47R1cWx2j8MScR1_/view?usp=sharing"> cm_10m </a>
  </td>
  <td width="20%" align="center"> 
  <a href="https://github.com/jzhoubu/HLP/blob/master/conf/train/pretrain_8xV100.yaml"> pretrain_8xV100 </a> 
  </td>
  <td width="20%" align="center"> 418M </td>
  <td width="20%" align="center"> 42.5 </td>
  <td width="20%" align="center"> 62.2 </td>
  <td width="20%" align="center"> 77.9 </td>
  <td width="20%" align="center"> 63.2 </td>
  <td width="20%" align="center"> 75.8 </td>
  <td width="20%" align="center"> 83.7 </td>
  <td width="20%" align="center"> 45.4 </td>
  <td width="20%" align="center"> 64.5 </td>
  <td width="20%" align="center"> 78.9 </td>
</tr>  


<tr>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/d/1-51Z-8li8IEDAeDjoyd2hUj_rwFhWc88/view?usp=sharing"> HLP </a> 
  </td>
  <td width="10%" align="center"> 
    <a href="https://drive.google.com/file/d/10YIohcsXAHKFzF2L43qkxH5zYkzTw70R/view?usp=sharing"> dl_10m </a>
    <a href="https://drive.google.com/file/d/10YWz5WN_qJAXVCON47R1cWx2j8MScR1_/view?usp=sharing"> cm_10m </a>
  </td>
  <td width="20%" align="center"> 
  <a href="https://github.com/jzhoubu/HLP/blob/master/conf/train/pretrain_8xV100.yaml"> pretrain_8xV100 </a> 
  </td>
  <td width="20%" align="center"> 418M </td>
  <td width="20%" align="center"> 50.9 </td>
  <td width="20%" align="center"> 69.3 </td>
  <td width="20%" align="center"> 82.1 </td>
  <td width="20%" align="center"> 65.3 </td>
  <td width="20%" align="center"> 77.0 </td>
  <td width="20%" align="center"> 84.1 </td>
  <td width="20%" align="center"> 49.1 </td>
  <td width="20%" align="center"> 67.4 </td>
  <td width="20%" align="center"> 80.5 </td>
</tr>

<tr>
  <td width="10%" colspan="1" align="center"> <b>Models</b> </td>
  <td width="10%" colspan="1" align="center"> <b>TuneSet</b> </td>
  <td width="10%" colspan="1" align="center"> <b>TuneConfig</b></td>
  <td width="10%" colspan="1" align="center"> <b>Size</b> </td>
  <td width="10%" colspan="9" align="center"> <b>Finetune Performance</b>  </td>
</tr>

<tr>
  <td width="10%" align="center">  HLP </td>
  <td width="10%" align="center"> 
  <a href="https://drive.google.com/file/u/4/d/1-3fy6UcjVJLt6CW7vRp_OkWb37WMBRBR/view?usp=sharing"> nq-train </a>
  
  </td>
  <td width="20%" align="center"> 
  <a href="https://github.com/jzhoubu/HLP/blob/master/conf/train/finetune_8xV100.yaml"> finetune_8xV100 </a> 
  </td>
  <td width="20%" align="center"> 840M </td>
  <td width="20%" align="center"> 70.6 </td>
  <td width="20%" align="center"> 81.3 </td>
  <td width="20%" align="center"> 88.0 </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
  <td width="20%" align="center"> / </td>
</tr>



</tbody>
</table>





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
