# Introduction

Here we present the instructions to reproduce the machine translation results from our ICML 2020 paper [PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/pdf/2003.07845.pdf), [video](https://drive.google.com/file/d/1M8spzzqNHAgNbdRcOJMKpbJ7GWj2y5mh/view?usp=sharing). The PowerNorm is implemented [here](https://github.com/amirgholami/powernorm/blob/master/fairseq/modules/norms/mask_powernorm.py). 

![](https://github.com/sIncerass/powernorm/blob/master/imgs/PN_LN_vis.png)

Here is the illustration plot of batch/power normalization (left) and layer normalization (right). The entries colored in blue show the components used for calculating the statistics.

The codes are based on open-sourced [fairseq](https://github.com/pytorch/fairseq) (v0.8.0). Follow [this link](https://fairseq.readthedocs.io/) for a detailed document about the original code base and [this link](https://github.com/pytorch/fairseq/tree/v0.8.0/examples/translation) for some examples of training baseline Transformer models for machine translation with fairseq.

We also provide [pre-trained models](#pre-trained-models) for several benchmark translation datasets.

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.7

The fairseq library we use requires PyTorch version >= 1.2.0.
Please follow the instructions [here](https://github.com/pytorch/pytorch#installation).

After PyTorch is installed, you can install fairseq with:
```
conda env create --file env.yml
python setup.py build develop
```

# Reproduction

The scripts for training and testing PowerNorm is located at `trans-scripts` folder. Please refer to [this page](trans-scripts/data-preprocessing/README.md) to preprocess and get binarized data or use the data we provided in the next section. To reproduce the results for Table.1 by yourself:

```
# IWSLT14 De-En
## To train the model
./trans-scripts/train/train-iwslt14.sh encoder_norm_self_attn encoder_norm_ffn decoder_norm_self_attn decoder_norm_ffn
example:
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh power power layer layer
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh batch batch layer layer
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/train/train-iwslt14.sh layer layer layer layer

## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/test/test-iwslt14.sh output_directory checkpoint_best.pt

# WMT14 En-De big
## To train the model, we are using 128 GPUs for our experiments.
./trans-scripts/train/train-wmt-big.sh encoder_norm_self_attn encoder_norm_ffn decoder_norm_self_attn decoder_norm_ffn
example:
$ CUDA_VISIBLE_DEVICES=0,1,2,3 ./trans-scripts/train/train-wmt14-big.sh power power layer layer

## To test a checkpoint
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/test/test-wmt14.sh output_directory checkpoint_best.pt
```

# Pre-trained Models

We provide following pre-trained models and pre-processed, binarized datasets for reproduction:

Description | Dataset | Model | Test set(s)
---|---|---|---
Transformer-PN `small` | [IWSLT14 German-English](https://drive.google.com/file/d/1fBG7DmbH0luD8EKqjviG5Equgkaxv3vv/view?usp=sharing) | [download (.tbz2)](https://drive.google.com/open?id=1aqOXAYnaEGhUmfyHHElFE-yL0c5uYg98) | IWSLT14 test set (shared vocab): <br> [download (.tbz2)](https://drive.google.com/open?id=1Vza4Yh7ev1336fWpgxGalkSLhb5dHxBa)

Example usage:
```
# IWSLT14 De-En
## at trans-net/translation/, after download the tbz2 file
$ tar xf powernorm_pretrain_iwslt.tbz2 
$ OUTPUT_DIR=iwslt14_de_en/powernorm_pretrain_iwslt
$ CKPT=averaged_model.pt
$ CUDA_VISIBLE_DEVICES=0 ./trans-scripts/test/test-iwslt14.sh $OUTPUT_DIR $CKPT
...
| Generate test with beam=5: BLEU4 = 35.87, 69.5/44.2/30.1/20.9 (BP=0.961, ratio=0.962, syslen=126196, reflen=131156)
```

## Citation
PowerNorm has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the library useful for your work:
```
@inproceedings{shen2020powernorm,
  title={PowerNorm: Rethinking Batch Normalization in Transformers},
  author={Shen, Sheng and Yao, Zhewei and Gholami, Amir and Mahoney, Michael and Keutzer, Kurt},
  booktitle={ICML},
  year={2020}
}
```
