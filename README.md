# Look Closer to Segment Better: Boundary Patch Refinement for Instance Segmentation (CVPR 2021)

## Introduction

PBR is a conceptually simple yet effective post-processing refinement framework to improve the boundary quality of instance segmentation. Following the idea of looking closer to segment boundaries better, BPR extracts and refines a series of small boundary patches along the predicted instance boundaries. The proposed BPR framework (as shown below) yields significant improvements over the Mask R-CNN baseline on the Cityscapes benchmark, especially on the boundary-aware metrics. 


<p align="center">
<img src="framework.png" width="80%" alt="framework"/>
</p>

For more details, please refer to our [paper](https://arxiv.org/abs/2104.05239).

## Installation

Please refer to [INSTALL.md](docs/install.md).


## Training

In this and the next section, we introduce how to train and inference BPR on the Cityscapes dataset. 
If you want to apply it to a COCO-like dataset, please refer to the next section.

We assume that the Cityscapes dataset is placed as follows:
```
BPR
├── data
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
```



### Prepare patches dataset [optional]

First, you need to generate the instance segmentation results on the Cityscapes training and validation set, as the following format:

```
maskrcnn_train
- aachen_000000_000019_leftImg8bit_pred.txt
- aachen_000001_000019_leftImg8bit_0_person.png
- aachen_000001_000019_leftImg8bit_10_car.png
- ...

maskrcnn_val
- frankfurt_000001_064130_leftImg8bit_pred.txt
- frankfurt_000001_064305_leftImg8bit_0_person.png
- frankfurt_000001_064305_leftImg8bit_10_motorcycle.png
- ...
```

The content of the txt file is the same as the standard format required by [cityscape script](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py), e.g.:

```
frankfurt_000000_000294_leftImg8bit_0_person.png 24 0.9990299940109253
frankfurt_000000_000294_leftImg8bit_1_person.png 24 0.9810258746147156
...
```

Then use the provided script to generate the training set:

```
sh tools/prepare_dataset.sh \
  maskrcnn_train \
  maskrcnn_val \
  maskrcnn_r50
```
Note that this step can take about 2 hours. Feel free to skip it by downloading the [processed training set](https://cloud.tsinghua.edu.cn/f/ea643dc32f824dbba28a/?dl=1).


### Train the network

Point `DATA_ROOT` to the patches dataset and run the training script 

```
DATA_ROOT=maskrcnn_r50/patches \
bash tools/dist_train.sh \
  configs/bpr/hrnet18s_128.py \
  4
```


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

The refinement results will be saved in `maskrcnn_val_refined/refined`.


## On other datasets

We also provide training and inference scripts suitable for the COCO dataset.
For those who want to apply BPR to their own datasets, we recommend converting them to the COCO format first.

We assume that the folder structure of the COCO data set is as follows:

```
BPR
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

### Training

First, a binary segmentation dataset needs to be constructed for training and validation of the Refinement Network.

This step requires coarse segmentation results (can come from any instance segmenter) on the training set and test set of COCO. Assuming that these two files are `mask_rcnn_r50.train.segm.json` and `mask_rcnn_r50.val.segm.json`, you only need to execute the following commands:

```
IOU_THRESH=0.15 \
sh tools/prepare_dataset_coco.sh \
  mask_rcnn_r50.train.segm.json \
  mask_rcnn_r50.val.segm.json \
  maskrcnn_r50 \
  70000
```

The dataset will be saved in `maskrcnn_r50/patches`. 

`IOU_THRESH=0.15` is used to control the threshold of nms.

The last argument (70000) means that only 70000 instances are sampled as the training set, since there are too many instances in the COCO dataset.
If you find that your computer cannot hold so many patches, you can try to reduce this value (may harm performance). 

In our paper, we used these two values for COCO dataset.

\
After building the dataset, use the following commands to train the Refinement Network:

```
DATA_ROOT=maskrcnn_r50/patches \
bash tools/dist_train.sh \
  configs/bpr/hrnet18s_128.py \
  4
```

### Inference

Use the following command to run inference:

```
IOU_THRESH=0.25 \
IMG_DIR=data/coco/val2017 \
GT_JSON=data/coco/annotations/instances_val2017.json \
GPUS=4 \
sh tools/inference_coco.sh \
  configs/bpr/hrnet18s_128.py \
  hrnet18s_coco-c172955f.pth \
  mask_rcnn_r50.val.segm.json \
  mask_rcnn_r50.val.refined.json
```

`IOU_THRESH` means the threshold of nms (see our paper for details). 

`IMG_DIR` and `GT_JSON` indicate the image folder and ground truth json file of COCO dataset.

`configs/bpr/hrnet18s_128.py` and `hrnet18s_coco-c172955f.pth` indicate the config file and checkpoint of Refinement Network.

`mask_rcnn_r50.val.segm.json` is the coasre instance segmentation results to be refined.

`mask_rcnn_r50.val.refined.json` saved the refined results.



## Models

| Backbone | Dataset | Config | Checkpoint |
| :------: | :------: | :------: | :------: |
| HRNet-18s | Cityscapes | [hrnet18s_128.py](configs/bpr/hrnet18s_128.py) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/a15da4d679654111ba89/?dl=1) |
| HRNet-48 | Cityscapes | [hrnet48_256.py](configs/bpr/hrnet48_256.py) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/54d7c737540444b38b18/?dl=1) |
| HRNet-18s | COCO | [hrnet18s_128.py](configs/bpr/hrnet18s_128.py) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/342fae1311b748a8b396/?dl=1) |

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
