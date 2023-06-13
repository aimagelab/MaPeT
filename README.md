<div align="center">
  <h1>MaPeT: Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training </h1>
</div>

[**Lorenzo Baraldi**](https://aimagelab.ing.unimore.it/imagelab/person.asp?idpersona=163)**&ast;**,
[**Roberto Amoroso**](https://scholar.google.com/citations?user=ZwnSLF8AAAAJ)**&ast;**,
[**Marcella Cornia**](https://scholar.google.com/citations?user=DzgmSJEAAAAJ),
[**Lorenzo Baraldi**](https://scholar.google.com/citations?user=V4RuMvsAAAAJ),
[**Andrea Pilzer**](https://scholar.google.com/citations?user=zooORRsAAAAJ),
[**Rita Cucchiara**](https://scholar.google.com/citations?user=OM3sZEoAAAAJ)

**&ast;** Equal contribution.

This is the **official repository** for the paper [Learning to Mask and Permute Visual Tokens for Vision Transformer Pre-Training]().

## Overview

<p align="center">
    <img src="images/model.png" style="max-width:800px">
</p>

>**Abstract**: <br>
> The use of self-supervised pre-training has emerged as a promising approach to enhance the performance of visual tasks such as image classification. In this context, recent approaches have employed the Masked Image Modeling paradigm, which pre-trains a backbone by reconstructing visual tokens associated with randomly masked image patches. This masking approach, however, introduces noise into the input data during pre-training, leading to discrepancies that can impair performance during the fine-tuning phase. Furthermore, input masking neglects the dependencies between corrupted patches, increasing the inconsistencies observed in downstream fine-tuning tasks. To overcome these issues, we propose a new self-supervised pre-training approach, named Masked and Permuted Vision Transformer (**MaPeT**), that employs autoregressive and permuted predictions to capture intra-patch dependencies. In addition, **MaPeT** employs auxiliary positional information to reduce the disparity between the pre-training and fine-tuning phases. In our experiments, we employ a fair setting to ensure reliable and meaningful comparisons and conduct investigations on multiple visual tokenizers, including our proposed _k_-CLIP which directly employs discretized CLIP features. Our results demonstrate that **MaPeT** achieves competitive performance on ImageNet, compared to baselines and competitors under the same model setting.

## Getting Started

Follow these steps to get started with the project:

1. Create a new conda environment: `conda create -n mapet python=3.8.16`
2. Activate the environment: `conda activate mapet`
3. Change directory to the project root: `cd MaPeT`
4. Install the required dependencies: `pip install -r requirements.txt`

To run validation with the default parameters, run the following command:

```python -u validate.py <PathToImageNet> --model <model_name> --checkpoint <checkpoint_path> --interpolation "bicubic" --amp --gp "avg" --pin-mem --no-prefetcher ```

where:
- ```<PathToImageNet>``` is the path to the ImageNet dataset (_e.g._, `data/ImageNet/ILSVRC/Data/CLS-LOC`).
- ```<model_name>``` is the name of the model to be used. Available models are:
    - ```vit_standard_tiny_patch16_224```: checkpoints for [k-CLIP](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_tiny_patch16_224_KCLIP.tar) and [VQ-KD](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_tiny_patch16_224_VQKD.tar).
    - ```vit_standard_small_patch16_224```: checkpoints for [k-CLIP](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_small_patch16_224_KCLIP.tar) and [VQ-KD](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_small_patch16_224_VQKD.tar).
    - ```vit_standard_base_patch16_224```: checkpoints for [k-CLIP](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_base_patch16_224_KCLIP.tar) and [VQ-KD](https://ailb-web.ing.unimore.it/publicfiles/MaPeT_checkpoints/vit_standard_base_patch16_224_VQKD.tar).
- ```<checkpoint_path>``` is the path to the checkpoint to be used.

## External Code employed

This project makes use of code on the following external code repositories:

- [TIMM](https://github.com/huggingface/pytorch-image-models): A collection of PyTorch models for computer vision tasks.
- [BEiT](https://github.com/microsoft/unilm/tree/master/beit2): BEiT implementation in Pytorch.
- [CAE](https://github.com/lxtGH/CAE): Code repository for the Contrastive Adversarial Exemplar (CAE) model.
- [CLIP](https://github.com/openai/CLIP): OpenAI's CLIP (Contrastive Language-Image Pretraining) model.
- [Faiss](https://github.com/facebookresearch/faiss): Faiss is a library for efficient similarity search and clustering of dense vectors.

## TODO
- [x] model weights
- [x] inference code
- [ ] training code


## Acknowledgements
We thank CINECA for providing computational resources. This work has partially been supported by the projects PNRR-M4C2 (PE00000013) "FAIR - Future Artificial Intelligence Research" funded by the European Commission and "ELSA - European Lighthouse on Secure and Safe AI" funded by the EU (GA 101070617).