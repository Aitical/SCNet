<div align="center">

# Fully 1x1 Convolutional Network for Lightweight Image Super-Resolution

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Kui Jiang](https://github.com/kuijiang94), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology.

---


[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2307.16140)
[![pretrained weights](https://img.shields.io/badge/Models-GoogleDrive-yellow.svg)](https://drive.google.com/drive/folders/1eUqL_8a9DQXZ2uCVyKeWB-6fO1ZdJciG?usp=sharing)
[![pretrained weights](https://img.shields.io/badge/Models-BaiduNetdisk-blue.svg)](https://pan.baidu.com/s/13_syaIXmG3lVnoMgzOS2Ag?pwd=SCSR)
[![visitors](https://hits.sh/github.com/Aitical/SCNet.svg)](https://hits.sh/github.com/Aitical/SCNet/)
</div>

This repository is the official PyTorch implementation of "Fully 1×1 Convolutional Network for Lightweight Image Super-Resolution". If our work helps your research or work, please cite it.
```
@article{wu2023fully,
    title={Fully $1\times1$ Convolutional Network for Lightweight Image Super-Resolution},
    author={Gang Wu and Junjun Jiang and Kui Jiang and Xianming Liu},
    year={2023},
    journal={Machine Intelligence Research},
    doi={10.1007/s11633-024-1401-z},
}
```
>Wu, Gang, Junjun Jiang, Kui Jiang and Xianming Liu. “Fully 1×1 Convolutional Network for Lightweight Image Super-Resolution.” Machine Intelligence Research.

## News

- [x] Update implementation codes.

- [x] Upload pre-trained weights utilized in manuscript. You can download from [Google Drive](https://drive.google.com/drive/folders/1eUqL_8a9DQXZ2uCVyKeWB-6fO1ZdJciG?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/13_syaIXmG3lVnoMgzOS2Ag?pwd=SCSR) with password `SCSR`.

## Overview
>Deep models have achieved significant process on single image super-resolution (SISR) tasks, in particular large models with large kernel (3×3 or more). However, the heavy computational footprint of such models prevents their deployment in real-time, resource-constrained environments. Conversely, 1×1 convolutions bring substantial computational efficiency, but struggle with aggregating local spatial representations, an essential capability to SISR models. In response to this dichotomy, we propose to harmonize the merits of both 3×3 and 1×1 kernels, and exploit a great potential for lightweight SISR tasks. Specifically, we propose a simple yet effective fully 1×1 convolutional network, named Shift-Conv-based Network (SCNet). By incorporating a parameter-free spatial-shift operation, it equips the fully 1×1 convolutional network with powerful representation capability while impressive computational efficiency. Extensive experiments demonstrate that SCNets, despite its fully 1×1 convolutional structure, consistently matches or even surpasses the performance of existing lightweight SR models that employ regular convolutions.

<div style="text-align: center">
<img src="https://s1.imagehub.cc/images/2024/03/23/3e9af849704577212b7b66b12bf7e990.png" alt="overview_SCNet.png" border="20" />
</div>


## Train

All experiments are evaluated based on [BasicSR](https://github.com/XPixelGroup/BasicSR), and we provide a minimal implementation in `SCNet_arch.py`.

For training, you may refer to the following script:
```
python basicsr/train.py -opt options/train/SCNet/SCNet-T-x4.yml
```
And for testing:
```
python basicsr/test.py -opt options/test/SCNet/SCNet-T-x4.yml
```
## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only. Please note that any commercial use of this code requires formal permission prior to use.

## Acknowledgement
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR). Thanks for their nice sharing.




