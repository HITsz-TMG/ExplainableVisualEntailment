# Chunk-aware Alignment and Lexical Constraint for Visual Entailment with Natural Language Explanations

## Introduction

This repository contains source code necessary to reproduce the results presented in the paper [Chunk-aware Alignment and Lexical Constraint for Visual Entailment with Natural Language Explanations](https://arxiv.org/abs/2207.11401).
We propose a unified Chunk-aware Alignment and Lexical Constraint based method, dubbed as CALeC for Visual Entailment with Natural Language Explanations. For more details, please refer to the paper.
We conduct extensive experiments on three datasets, and experimental results indicate that CALeC significantly outperforms other competitor models on inference accuracy and quality of generated explanations.

![avatar](model.png)

## Performance

Follow [e-ViL](https://github.com/maximek3/e-ViL), we test our model on e-SNLI-VE, VQA-X and VCR. Please refer to the benchmark repository for the datasets detail.
We only list $S_O$, $S_T$ and $S_E$ there, please refer to our [paper](https://arxiv.org/abs/2207.11401) for detailed results.

**e-SNLI-VE**

Model   | $S_O$    | $S_T$ | $S_E$   |
--------|-----|--------|-------|
PJ-X    |  20.40 | 0.998  | 0.999 | 
FME | 0.996 | 0.998  | 0.997 | 
RVT  | 1.0 | 1.0    | 1.0   | 
e-UG<sup>*</sup>   | 1.0 | 1.0    | 1.0   | 
CALeC | 0.998 | 1.0    | 0.999 | 
NLX-GPT<sup>†</sup>   | 0.996 | 1.0    | 0.998 | 
CALeC<sup>†</sup> | 1.0 | 1.0    | 1.0   | 

## Training procedure

We conduct experiments on three datasets: VQA-X, e-SNLI-VE and VCR. These datasets can be downloaded from [e-vil](https://github.com/maximek3/e-ViL).
We extract the corresponding image features using [VilVL](https://github.com/pzzhang/VinVL) and pack them into *pkl* file.
The *pkl* file contains: 



Here is an example of get the borders of each chunk.

```
python ./utils/GetChunk_v4_SNLI.py 
```

The border index will be saved as a dictionary in .pkl.

##Training
Here is an example to pre-train CSI on Flickr30k:

```
python CSI_prertain_align_only.py --do_train --do_lower_case --save_steps 1000 --output_dir ./outputs/CSI_pre_train
```

Download [cococaption](https://github.com/tylin/coco-caption) for evaluation.
Here is an example to train the model on e-SNLI-VE:

```
python run_SNLI_CALEC.py --do_train --do_lower_case --save_steps 1000 --output_dir ./outputs/SNLI
```

The checkpoints will be saved in the output_dir

## Testing

Here is an example to test a trained model on the e-SNLI-VE test set:

```
python run_SNLI_CALEC_CBS.py --do_test --do_lower_case --eval_model_dir your_save_checkpoint_path --constrained 0.86
```

The --constrained is the constrained coefficient used in constrained beam sample.
All generated explanations and a text log will be saved in the given output directory (*your_save_checkpoint_path*).

## COCOcaption package for automatic NLG metrics

In order to run NLG evaluation in this code you need to download the package from this Google Drive link. It needs to be placed in the root directory of this project.


## Framework versions

* Pytorch 1.7.1+cu110
* Transformers 4.18.0

## Citations

Please consider citing this paper if you use the code:


```
@inproceedings{yang2022chunk,
  title={Chunk-aware Alignment and Lexical Constraint for Visual Entailment with Natural Language Explanations},
  author={Yang, Qian and Li, Yunxin and Hu, Baotian and Ma, Lin and Ding, Yuxin and Zhang, Min},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={3587--3597},
  year={2022}
}

```