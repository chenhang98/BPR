import sys
import os
import os.path as osp
import cv2
import json
import numpy as np
from tqdm import *
from functools import partial
import multiprocessing as mp
from collections import defaultdict
from pycocotools import mask as maskUtils
from fire import Fire


def load_info(gt_json):
    gt = json.load(open(gt_json))
    imgid2fn = dict()
    for item in gt['images']:
        imgid2fn[item['id']] = osp.basename(
            item['file_name']).split('.')[0]
    catid2name = dict()
    for item in gt['categories']:
        catid2name[item['id']] = item['name']
    return imgid2fn, catid2name


def load_dt(dt_json):
    dt = json.load(open(dt_json))
    imgid2dt = defaultdict(list)
    for item in dt:
        imgid2dt[item['image_id']].append(item)
    return imgid2dt


def worker(imgid, imgid2dt, imgid2fn, catid2name, out_dir):
    dts = imgid2dt[imgid]
    fn = imgid2fn[imgid]
    with open(osp.join(out_dir, f'{fn}_pred.txt'), 'w') as f:
        for instid, item in enumerate(dts):
            cat_id = item['category_id']
            score = item['score']
            cat_name = catid2name[cat_id]
            pngfn = f'{fn}_{instid}_{cat_name}.png'
            mask = maskUtils.decode(item['segmentation'])
            cv2.imwrite(osp.join(out_dir, pngfn), mask)
            f.write(f'{pngfn} {cat_id} {score}\n')


def main(dt_json, gt_json, out_dir):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    imgid2fn, catid2name = load_info(gt_json)
    imgid2dt = load_dt(dt_json)

    with mp.Pool(processes=20) as p:
        with tqdm(total=len(imgid2fn)) as pbar:
            for _ in p.imap_unordered(
                    partial(worker, imgid2dt=imgid2dt,
                            imgid2fn=imgid2fn,
                            catid2name=catid2name,
                            out_dir=out_dir),
                    imgid2fn.keys()):
                pbar.update()


if __name__ == '__main__':
    Fire(main)
