import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from pycocotools.coco import COCO
import multiprocessing as mp
from fire import Fire


coco = None


def cal_iou(mask1, mask2):
    si = np.sum(mask1 & mask2)
    su = np.sum(mask1 | mask2)
    return si / su


def max_iou(inst):
    imgid = inst['image_id']
    catid = inst['category_id']
    maskdt = coco.annToMask(inst)

    masks = []
    annids = coco.getAnnIds(imgIds=imgid, catIds=catid)
    anns = coco.loadAnns(annids)
    for ann in anns:
        if not ann.get('iscrowd', False):
            masks.append(coco.annToMask(ann))

    ious = [cal_iou(maskdt, _) for _ in masks]
    miou = max(ious) if len(ious) else 0
    return miou, inst


def filter_iou(bboxs, thresh=0.5):
    out = list()
    with mp.Pool(processes=20) as p:
        with tqdm(total=len(bboxs)) as pbar:
            for iou, bbox in p.imap(max_iou, bboxs):
                if iou > thresh:
                    out.append(bbox)
                pbar.update()
    return out


def main(dt_json, gt_json, out_json, thresh=0.5):
    """Filter out instances with IoU < thresh.

    Args:
        dt_json (str): path to instance segmentation's results.
        gt_json (str): path to ground truth.
        out_json (str): path to save the results.
        thresh (float, optional): IoU threshold. Defaults to 0.5.
    """
    global coco
    coco = COCO(gt_json)
    dt = json.load(open(dt_json))
    dt_filtered = filter_iou(dt, thresh)
    with open(out_json, 'w') as f:
        json.dump(dt_filtered, f)


if __name__=='__main__':
   Fire(main)
