# Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (CVPR 2021)



## Introduction

PBR is a conceptually simple yet effective post-processing refinement framework to improve the boundary quality of instance segmentation. Following the idea of looking closer to segment boundaries better, BPR extracts and refines a series of small boundary patches along the predicted instance boundaries. The proposed BPR framework (as shown below) yields significant improvements over the Mask R-CNN baseline on the Cityscapes benchmark, especially on the boundary-aware metrics. 


<p align="center">
<img src="framework.png" width="80%" alt="framework"/>
</p>

For more details, please refer to our [paper](https://arxiv.org/abs/2104.05239).

## Installation

Please refer to [INSTALL.md](docs/install.md).


## Inference

Suppose you have some instance segmentation results of Cityscapes dataset, as the following format:

```
maskrcnn_val
- frankfurt_000001_064130_leftImg8bit_pred.txt
- frankfurt_000001_064305_leftImg8bit_0_person.png
- frankfurt_000001_064305_leftImg8bit_10_motorcycle.png
- ...
```

We provide a script ([tools/inference.sh](tools/inference.sh)) to perform refinement operation, usage:

```
IOU_THRESH=0.55 \
IMG_DIR=data/cityscapes/leftImg8bit/val \
GT_JSON=data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json \
BPR_ROOT=. \
GPUS=4 \
sh tools/inference.sh configs/bpr/hrnet48_256.py ckpts/hrnet48_256.pth maskrcnn_val maskrcnn_val_refined
```

The refinement results will be saved in `maskrcnn_val_refined/refined`


## Models

| Backbone | Dataset | Checkpoint |
| :------: | :------: | :------: |
| HRNet-18s | Cityscapes | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a15da4d679654111ba89/?dl=1) |
| HRNet-48 | Cityscapes | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/54d7c737540444b38b18/?dl=1) |

## Acknowledgement

This project is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) code base.

## Citation

If you find this project useful in your research, please consider citing:

```
@article{tang2021look,
  title={Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation},
  author={Chufeng Tang and Hang Chen and Xiao Li and Jianmin Li and Zhaoxiang Zhang and Xiaolin Hu},
  journal={arXiv preprint arXiv:2104.05239},
  year={2021}
}
```
