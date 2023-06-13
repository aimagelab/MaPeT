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


## TODO
- [ ] model weights
- [ ] inference code
- [ ] training code


## Acknowledgements
We thank CINECA for providing computational resources. This work has partially been supported by the projects PNRR-M4C2 (PE00000013) "FAIR - Future Artificial Intelligence Research" funded by the European Commission and "ELSA - European Lighthouse on Secure and Safe AI" funded by the EU (GA 101070617).