# kNN-transformers: Nearest-Neighbor Language Models based on Hugging Face's 🤗 `transformers` library

This is a Hugging Face's 🤗 `transformers` implementation of k-nearest-neighbor-based language models,
designed to be easy and useful in research, and for experimenting with new ideas in kNN-based models. 

All previous kNN-LM based implementations are implemented in the `fairseq` library, and **they duplicated the library's entire codebase** to implement their modification.
These include the official kNN-LM repository [https://github.com/urvashik/knnlm](https://github.com/urvashik/knnlm), the official RetoMaton repository [https://github.com/neulab/retomaton](https://github.com/neulab/retomaton), and others.


We implements the [k-Nearest Neighbor Language Model](https://arxiv.org/pdf/1911.00172.pdf) (Khandelwal et al., ICLR'2020), and this is also
an official implementation of the RetoMaton model described in:
["Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval"](https://arxiv.org/pdf/2201.12431.pdf) (ICML'2022). Most importantly, we implement these models in 🤗 `transformers`, and without modifying the `transformers` library itself.

To use this repository, all you need to do is copy its `*.py` files into your project.
You can load any language model from Hugging Face's hub (such as `gpt2`, by `model = AutoModelForCausalLM.from_pretrained(...)`, build a datastore or download ours (need to be performed only once), and then:
```python
knn_wrapper = KNNWrapper(...) # or: RetomatonWrapper(...)
knn_wrapper.break_into(model)
```

That's it! The model now internally uses kNN-LM or RetoMaton. 

The files `knnlm.py` and `retomaton.py` are standalone and can be copied to any project. The file `run_clm.py` is an example that shows how to load and run kNN-LM and RetoMaton.


Please let us know if anythings is not working as expected, and feel free to create [new issues](https://github.com/neulab/knn-transformers/issues) with any questions.


Table of Contents
=================
  * [Background](#background)
  * [Results](#results)
  * [Available Models](#available-models)
  * [Quickstart](#quickstart)
    * [Requirements](#requirements)
    * [Step 1: Evaluating the base Language Model](#step-1-evaluating-the-base-language-model)
    * [Step 2: Saving a Datastore](#step-2-saving-a-datastore)
    * [Step 4: Saving the keys and values for the datastore](#step-4-saving-the-keys-and-values-for-the-datastore)
    * [Step 5: Building the FAISS index](#step-5-building-the-faiss-index)
    * [Step 6: Evaluating RetoMaton without clustering](#step-6-evaluating-retomaton-without-clustering)
    * [Step 7: Adding clustering](#step-7-adding-clustering)
    * [Step 8: Evaluating the Fine-tuned Model](#step-8-evaluating-the-fine-tuned-model)
  * [Lambda values](#lambda-values)
  * [All files](#all-files)
  * [Differences from the kNN-LM implementation](#differences-from-the-knn-lm-implementation)
  * [Citation](#citation)

## Background

### kNN-LM
The k-Nearest Neighbor Language Model takes an already-trained model, performs a single forward pass over the entire training set, and creates a datastore of `(key,value)` pairs, where `key` is a hidden representation of the trained model after reading a training example, and `value` is the token that should be predicted next.

At test time, for every predicted token, the model performs a k-nearest neighbors search in the datastore, retrieves the `(key,value)` pairs that are closest to the test hidden representation, and normalizes their distances using softmax. Finally, the model interpolates the base LM's probability with the probability formed by the retrieved nearest neighbors and their normalized distances.

For more details, see the [paper by Khandelwal et al., ICLR'2020](https://arxiv.org/pdf/1911.00172.pdf)

### RetoMaton
RetoMaton extends kNN-LM, by (1) saving a *pointer* in every datastore entry; and (2) clustering entries according to their keys. That is, every datastore entry is now a tuple `(key,value,pointer)`, and it belongs to a cluster. 

These two changes create an automaton from the datastore, where states are clusters, edges are pointers (shared among examples in the same cluster), and transition weights are the normalized distances between the test representation and each key.

At test time, the model traverses the automaton, and follows the edges according to the token that was predicted.
This allows to save up to 80% of the kNN searches by following pointers instead of performing the expensive search, or reducing perplexity without saving searches.

For more details, see the [paper by Alon et al., ICML'2022](https://arxiv.org/pdf/2201.12431.pdf)

<center style="padding: 40px"><img width="60%" src="images/overview.jpeg" /></center>

## Results - **Wikitext-103**
The exact results from the RetoMaton papers can be reproduced using the code at [https://github.com/neulab/retomaton](https://github.com/neulab/retomaton) (based on `fairseq`).

The following results were obtained using the code in this repository:



| Base LM:        | `distilgpt2` | `gpt2` | 
| :---            |    ----:   |     ---: |
| base perplexity | 18.25      | 14.84    |
| kNN-LM          |  15.03     |   12.57  |
| RetoMaton       | **14.70**  |  **12.46**    |

And when varying the fraction of saved searches:
TODO


These are the results from the RetoMaton paper, on a model that was trained on Wikitext-103 from scratch:
<center style="padding: 40px"><img width="60%" src="images/wiki.png" /></center>


## Available Models

kNN-LM and RetoMaton datastores depend on the LM that was used to create them. We fine-tuned a few `gpt2`-based models on the training set of Wikitext-103 (because Wikitext-103 was not included in GPT2's pretraining data):
* `neulab/distilgpt2-finetuned-wikitext103`
* `neulab/gpt2-finetuned-wikitext103`
* `neulab/gpt2-med-finetuned-wikitext103`
* `neulab/gpt2-large-finetuned-wikitext103`

All these models are available at the Hugging Face Hub and can be loaded by (for example):
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained('neulab/gpt2-finetuned-wikitext103')
model = AutoModelForCausalLM.from_pretrained('neulab/gpt2-finetuned-wikitext103')
```

This project is not limited to these models, and can work with any language model.
We fine-tuned all models using:
```bash
python run_clm.py --model_name_or_path <base_model_name> \
    --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
    --do_train --do_eval --output_dir finetune_gpt2_wikitext103/ \
    --save_total_limit 5 --per_device_train_batch_size 2
```
Where `<base_model_name>` is, for example, `gpt2`, `distilgpt2`, `gpt2-med`, `gpt2-large`, or `gpt2-xl`.

## Quickstart

### Step 0: Clone this repository:
```bash
git clone https://github.com/neulab/knn-transformers
cd knn-transformers
```

#### Requirements
Run:
```bash
pip install requirements.txt`
```

* The project also depends on the `faiss` library. In MacOS, use the Anaconda installation instead:
```
conda install -c conda-forge faiss-cpu
```

### Step 1: Evaluating the base Language Model

To evaluate the fine-tuned model (for example, `neulab/gpt2-finetuned-wikitext103`) on the validation set (without any retrieval):

```bash
MODEL=neulab/gpt2-finetuned-wikitext103

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation
```

### Step 2: Saving a Datastore

You can either download our preprocessed Wikitext-103 datastores, or preprocess them yourself.

To download a datastore for Wikitext-103 that we created for the finetuned `gpt2` model (`neulab/gpt2-finetuned-wikitext103`):
```bash
wget -P checkpoints/gpt2/ https://knn-transformers.s3.amazonaws.com/gpt2/dstore_gpt2_116988150_768_vals.npy
wget https://knn-transformers.s3.amazonaws.com/gpt2/dstore_gpt2_116988150_768_keys.npy
```

Similarly, we created datastores using the `distilgpt2-finetuned-wikitext103` model.
To see all available datastores, go to: [https://knn-transformers.s3.amazonaws.com/index.html](https://knn-transformers.s3.amazonaws.com/index.html)

To save a datastore, run:
```bash
MODEL=neulab/gpt2-finetuned-wikitext103

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_eval --eval_subset train \
  --output_dir checkpoints/${MODEL} \
  --dstore_size 116988150 --dstore_dir checkpoints/${MODEL} \
  --save_knnlm_dstore
```

A Wikitext-103 datastore requires about 200GB of disk space.

### Step 3: Building the FAISS index

The FAISS index requires a training stage where it learns an index for accessing the keys quickly. This step does not require a GPU.

To download our index:
TODO
```
wget -P checkpoints/gpt2/ https://knn-transformers.s3.amazonaws.com/gpt2/index_gpt2_116988150_768.indexed
```
 
To build the FAISS index yourself (not needed if you already downloaded ours):
```bash
MODEL=neulab/gpt2-finetuned-wikitext103

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_eval \
  --output_dir checkpoints/${MODEL} \
  --dstore_size 116988150 --dstore_dir checkpoints/${MODEL} \
  --build_index
```


### Step 4: Evaluating Models

To evaluate kNN-LM and RetoMaton on the validation set:

#### Wikitext-103:

```bash
MODEL=neulab/gpt2-finetuned-wikitext103

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_size 116988150 --dstore_dir checkpoints/${MODEL} --knn
```

To encourage the model to perform a full kNN search more frequently and thus increase accuracy and reduce perplexity, use a larger value of `--min-knns` such as `100`. Using `--min-knns 9999999` makes the model perform kNN search at every step (`FoSS = 0` in Figure 3 of the paper), and achieves the best results at the cost of slower speed.


### Step 7: Adding clustering

For the Greedy Merge clustering algorithm. See [the code of He et al. (2021)](https://github.com/jxhe/efficient-knnlm/blob/main/ef_knnlm/dstore_compression/greedy_merge.sh). Greedy Merge is much faster and requires much fewer memory than k-means, but results in slightly higher perplexity:



See also Figures 8 and 9 in Appendix D in the paper.

#### To download our clusters for Wikitext-103:
Note that only **one** of the following files is needed. For the main experiments in the paper, we used:
```bash
wget -P checkpoints/wt103/ https://retomaton.s3.us-east-2.amazonaws.com/wt103/clusters_s40000000_k1000000_members.pkl
```

but additional clusterings are available as well:
```bash
wget -P checkpoints/wt103/ https://retomaton.s3.us-east-2.amazonaws.com/wt103/clusters_s20000000_k500000_members.pkl
wget -P checkpoints/wt103/ https://retomaton.s3.us-east-2.amazonaws.com/wt103/dstore_merge15_members_sp.pkl
wget -P checkpoints/wt103/ https://retomaton.s3.us-east-2.amazonaws.com/wt103/dstore_merge29_members.pkl
```

#### To download our clusters for Law-MT:
Note that only **one** of the following files is needed. For the main experiments in the paper, we used:
```bash
wget -P checkpoints/law/ https://retomaton.s3.us-east-2.amazonaws.com/law/law_clusters_s40000000_k200000_members.pkl
```

but additional clustering is available as well:
```bash
wget -P checkpoints/law/ https://retomaton.s3.us-east-2.amazonaws.com/law/law_clusters_s40000000_k400000_members.pkl
```

#### Evaluating RetoMaton with clustering:
Basically identical to [Step 6: Evaluating RetoMaton without clustering](#step-6-evaluating-retomaton-without-clustering), except that we add the flag `--members <filename>_members.pkl`, 

##### Wikitext-103:

```bash
DSTORE=checkpoints/wt103/dstore16
DSTORE_SIZE=103225485
INDEX=checkpoints/wt103/knn16.index
MODEL=checkpoints/wt103/wt103_checkpoint_best.pt
MEMBERS=checkpoints/wt103/clusters_s40000000_k1000000_members.pkl

python eval_lm.py data-bin/wikitext-103 \
    --path ${MODEL} \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 1024000 --batch-size 2 \
    --gen-subset valid --dstore-filename ${DSTORE} \
    --indexfile ${INDEX}  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size ${DSTORE_SIZE} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --dstore-fp16 \
    --knn-sim-func do_not_recomp_l2 --no-load-keys --move-dstore-to-mem \
    --knnlm-gpu --min-knns 1 --max-knns 1024 \
    --members ${MEMBERS}
```

##### Law-MT:
```bash
DSTORE=checkpoints/law/dstore16
DSTORE_SIZE=19068709
INDEX=checkpoints/law/knn16.index
MODEL=checkpoints/law/wmt19.en/model.pt
MEMBERS=checkpoints/law/law_clusters_s40000000_k200000_members.pkl

python eval_lm.py data-bin/law \
    --path ${MODEL} \
    --sample-break-mode eos --max-tokens 2048 \
    --context-window 0 --softmax-batch 1024000 --batch-size 2 \
    --gen-subset valid --dstore-filename ${DSTORE} \
    --indexfile ${INDEX}  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.9 --dstore-size ${DSTORE_SIZE} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --dstore-fp16 \
    --knn-sim-func do_not_recomp_l2 --no-load-keys --move-dstore-to-mem \
    --remove-bpe \
    --knnlm-gpu --min-knns 1 --max-knns 1024\
    --members ${MEMBERS}
```


#### Cluster the keys yourself (not needed if you downloaded our clusters):
for Wikitext-103:
```
DSTORE=checkpoints/wt103/dstore16
DSTORE_SIZE=103225485
NUM_CLUSTERS=1000000
SAMPLE=40000000
DIM=1024
SAVE=kmeans_wt103
```

For Law-MT:
```bash
DSTORE=checkpoints/law/dstore16
DSTORE_SIZE=19068709
NUM_CLUSTERS=200000
SAMPLE=40000000
DIM=1536
SAVE=kmeans_law
```

And then for both datasets:
```bash
python kmeans.py --dstore ${DSTORE} --dstore-size ${DSTORE_SIZE} --num-clusters ${NUM_CLUSTERS} --sample ${SAMPLE} --dim ${DIM} --save ${}
```



## Step 8: Evaluating the Fine-tuned Model
The model that was fine-tuned on Law-MT, along with its corresponding datastore, FAISS index and clustering can be downloaded from:

```bash
mkdir checkpoints/law-finetuned/
wget -P checkpoints/law-finetuned/ https://retomaton.s3.us-east-2.amazonaws.com/law/finetuned.pt
wget -P checkpoints/law-finetuned/ https://retomaton.s3.us-east-2.amazonaws.com/law/dstore16_finetuned_size19068709_embed1536_fp16_vals.npy
wget -P checkpoints/law-finetuned/ https://retomaton.s3.us-east-2.amazonaws.com/law/dstore16_finetuned_size19068709_embed1536_fp16_keys.npy
wget -P checkpoints/law-finetuned/ https://retomaton.s3.us-east-2.amazonaws.com/law/knn_finetuned.index
wget -P checkpoints/law-finetuned/ https://retomaton.s3.us-east-2.amazonaws.com/law/law_finetuned_clusters_s20000000_k200000_members.pkl
```

Finally, [evaluate](#evaluating-retomaton-without-clustering) using the fine-tuned checkpoint, datastore, and index. 

**It is important** to also set `--lmbda 0.25` when using the fine-tuned model: since the model is fine-tuned, we can rely on it more than before. See a clarification at [#lambda-values](#lambda-values)

Best results with the fine-tuned model are achieved _without_ clustering (that is, every datastore entry is a singleton cluster).

Then, the same steps as before should be run on the Law-MT datasets, except that: 
* `finetuned.pt` should be used as the `${MODEL}`
* `dstore16_finetuned_size19068709_embed1536_fp16` should be used as the `${DSTORE}`
* `knn_finetuned.index` should be used as the `${INDEX}`
* `law_finetuned_clusters_s20000000_k200000_members.pkl` shoould be used as`${MEMBERS}`

That is:

```bash
DSTORE=checkpoints/law-finetuned/dstore16_finetuned_size19068709_embed1536_fp16
DSTORE_SIZE=19068709
INDEX=checkpoints/law-finetuned/knn_finetuned.index
MODEL=checkpoints/law-finetuned/finetuned.pt
MEMBERS=checkpoints/law-finetuned/law_finetuned_clusters_s20000000_k200000_members.pkl

python eval_lm.py data-bin/law \
    --path ${MODEL} \
    --sample-break-mode eos --max-tokens 2048 \
    --context-window 0 --softmax-batch 1024000 --batch-size 2 \
    --gen-subset valid --dstore-filename ${DSTORE} \
    --indexfile ${INDEX}  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size ${DSTORE_SIZE} --knn-keytype last_ffn_input \
    --probe 32 --knnlm --dstore-fp16 \
    --knn-sim-func do_not_recomp_l2 --no-load-keys --move-dstore-to-mem \
    --remove-bpe \
    --knnlm-gpu --min-knns 1 --max-knns 1024
```


## Lambda values
In all configurations, the interpolation factor `lmbda` is set to `0.25`, except when the base LM is `checkpoints/law/wmt19.en/model.pt` **and** the model is evaluated on Law-MT, since this scenario tests domain adaptation, and thus `lmbda` should be set to `0.9`:

|             | `wt103_checkpoint_best.pt` | `wmt19.en/model.pt`     | `finetuned.pt` |
| :---        |    ----:   |     ---: | ---: |
| Wikitext-103| 0.25       | -    |   -   |
| Law-MT      | -       | 0.9    |   0.25 |

## All files: 
Checkpoints and datasets can be downloaded from here:
[https://zenodo.org/record/6525426](https://zenodo.org/record/6525426)

And also from the [AWS S3 bucket](https://retomaton.s3.us-east-2.amazonaws.com/) 


## Differences from the kNN-LM implementation

### Implementation Pointers
Here we point to the code that differs our work from kNN-LM.
* The main changes are in this
[commit](https://github.com/neulab/retomaton/commit/89a29d1ac6e8c1360637aa1bfe77a1be227e83cc). The pointers for the next timestep are initially [the current k-nearest neighbors + 1](fairseq/sequence_scorer.py#L203). Then we extend each pointer to [consider all entries in its cluster](fairseq/sequence_scorer.py#L216). This is [the function](fairseq/sequence_scorer.py#L240-L251) that maps each pointer to its cluster, removes duplicate clusters, and then finds the members of each cluster. We  [find the log probabilities](fairseq/sequence_scorer.py#L218-L222) as suggested by the new pointers, and finally take to the next timestep - [only the pointers that are consistent](fairseq/sequence_scorer.py#L228) with the token that the model eventually predicted.
* In [this commit](https://github.com/neulab/retomaton/commit/99cb52001b3c87b15dd8ef892172cfac334bcef5) we [utilize the given pointers](fairseq/knnlm.py#L131-L133), or [perform kNN search](fairseq/knnlm.py#L131-L133) and combine the results with the existing pointers.
* When using the `--knnlm-gpu` flag, we use a [GPU index](fairseq/knnlm.py#L34-L38) to search for nearest neighbors, and its copy [CPU index](fairseq/knnlm.py#L43-L47) to reconstruct vectors given their ID. Unfortunately, currently reconstructing vectors in `faiss` is [not implemented for GPU indexes](https://github.com/facebookresearch/faiss/issues/2181) (see also [this issue](https://github.com/facebookresearch/faiss/issues/314)). 
* Reconstructing a **batch** of vectors from the index is unfortunately not implemented in `faiss` (see [this issue](https://github.com/facebookresearch/faiss/issues/1163)), and thus the fastest way that we found to do that is using `np.vectorize`, and reconstructing many single vectors in parallel: [fairseq/knnlm.py#L92-L94](fairseq/knnlm.py#L92-L94).
* Performing k-means clustering on millions of vectors can be performed in many ways, but specifically we utilize the `faiss` library to do it using the script [kmeans.py](kmeans.py).

### Additional minor differences:
* The original [kNN-LM](https://github.com/urvashik/knnlm) repository uses `faiss` CPU to perform retrieval. However, we added the flag `--knnlm-gpu` that allows performing retrieval much faster on the GPU.
* After each retrieval, the original [kNN-LM](https://github.com/urvashik/knnlm) repository loads the found keys and re-computes the distance from the query to each nearest neighbor. This is much more time consuming, unless loading all the keys (200GB) into memory.
We thus use the flags `--knn-sim-func do_not_recomp_l2 --no-load-keys --move-dstore-to-mem`.
* When using `faiss-gpu`, it is useful to [`import faiss.contrib.torch_utils`](fairseq/knnlm.py#L3). This allows performing the kNN search using `torch` tensors (rather than only `numpy` arrays). Additionally, sometimes this `import` statement prevents searching bugs in `faiss` (see [this issue](https://github.com/facebookresearch/faiss/issues/2126)).




## Citation
If you use this code for research, please cite:

[Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://arxiv.org/pdf/2201.12431.pdf)

```
@article{alon2022neuro,
  title={Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval},
  author={Alon, Uri and Xu, Frank F and He, Junxian and Sengupta, Sudipta and Roth, Dan and Neubig, Graham},
  journal={arXiv preprint arXiv:2201.12431},
  year={2022}
}
```

[Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/pdf/1911.00172)
```
@inproceedings{khandelwal20generalization,
  title={{Generalization through Memorization: Nearest Neighbor Language Models}},
  author={Khandelwal, Urvashi and Levy, Omer and Jurafsky, Dan and Zettlemoyer, Luke and Lewis, Mike},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2020}
}
```