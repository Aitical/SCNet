<div align="center">

# Fully 1x1 Convolutional Network for Lightweight Image Super-Resolution

[Gang Wu](https://scholar.google.com/citations?user=JSqb7QIAAAAJ), [Junjun Jiang](http://homepage.hit.edu.cn/jiangjunjun), [Kui Jiang](https://github.com/kuijiang94), and [Xianming Liu](http://homepage.hit.edu.cn/xmliu)

[AIIA Lab](https://aiialabhit.github.io/team/), Harbin Institute of Technology.

---
[arxiv](https://arxiv.org/abs/2307.16140)
**|**
[pretrained models]()


[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.16140)
[![visitors](https://hits.sh/github.com/Aitical/SCNet.svg)](https://hits.sh/github.com/Aitical/SCNet/)
</div>

This repository is the official PyTorch implementation of "Fully 1×1 Convolutional Network for Lightweight Image Super-Resolution"
>Deep models have achieved significant process on single image super-resolution (SISR) tasks, in particular large models with large kernel (3×3 or more). However, the heavy computational footprint of such models prevents their deployment in real-time, resource-constrained environments. Conversely, 1×1 convolutions bring substantial computational efficiency, but struggle with aggregating local spatial representations, an essential capability to SISR models. In response to this dichotomy, we propose to harmonize the merits of both 3×3 and 1×1 kernels, and exploit a great potential for lightweight SISR tasks. Specifically, we propose a simple yet effective fully 1×1 convolutional network, named Shift-Conv-based Network (SCNet). By incorporating a parameter-free spatial-shift operation, it equips the fully 1×1 convolutional network with powerful representation capability while impressive computational efficiency. Extensive experiments demonstrate that SCNets, despite its fully 1×1 convolutional structure, consistently matches or even surpasses the performance of existing lightweight SR models that employ regular convolutions.

## Train

All experiments are evaluated based on [BasicSR](https://github.com/XPixelGroup/BasicSR), and we provide a minimal implementation in `SCNet_arch.py`. Download and put it into `basicsr/archs`. 


## TODO
- [ ] Add pretrained model
- [ ] Add results of test set



## Citation
```
@misc{wu2023fully,
    title={Fully $1\times1$ Convolutional Network for Lightweight Image Super-Resolution},
    author={Gang Wu and Junjun Jiang and Kui Jiang and Xianming Liu},
    year={2023},
    eprint={2307.16140},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
