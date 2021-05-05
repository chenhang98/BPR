import os
import os.path as osp
import json
import numpy as np
import pickle as pkl
import multiprocessing as mp
from tqdm import tqdm
from argparse import ArgumentParser
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


coco = None
results = None


def run_inst(i_inst):
    inst_id, inst = i_inst
    global coco
    global results

    # load patch proposals
    path = osp.join(args.details_dir, f"{inst_id}.txt")
    with open(path) as f:
        info = eval(f.read())
    patches, hlist, wlist = info["patches"], info["hlist"], info["wlist"]

    # reassemble
    newmask = coco.annToMask(inst)
    newmask_refined = np.zeros_like(newmask, dtype=np.float32)
    newmask_count = np.zeros_like(newmask_refined)

    for j, pid in enumerate(patches):
        y, h = hlist[j]
        x, w = wlist[j]
        patch_mask = results[pid]
        if args.padding:
            p = args.padding
            patch_mask = patch_mask[p:-p, p:-p]
        newmask_refined[y:y+h, x:x+w] += patch_mask
        newmask_count[y:y+h, x:x+w] += 1

    s = newmask_count > 0
    newmask_refined[s] /= newmask_count[s]
    newmask[s] = newmask_refined[s] > 0.5

    # update
    newmask = newmask.astype(np.uint8)
    segm = mask_utils.encode(np.asfortranarray(newmask))
    inst["segmentation"]["counts"] = \
        segm["counts"].decode("utf8")

    return inst


def start():
    global coco
    global results
    coco = COCO(args.gt_json)
    dt = json.load(open(args.dt_json))

    # load network's outputs 
    with open(args.res_pkl, 'rb') as f:
        _res = pkl.load(f)   # img_infos, masks
    results = dict()
    for img_info, mask in zip(*_res):
        patch_id = int(img_info['ann']['seg_map'].split('.')[0])
        results[patch_id] = mask

    # reassemble with multi-process
    refined_res = []
    with mp.Pool(processes=args.num_proc) as p:
        with tqdm(total=len(dt)) as pbar:
            for r in p.imap_unordered(run_inst, enumerate(dt)):
                refined_res.append(r)
                pbar.update()

    with open(args.out_json, 'w') as f:
        json.dump(refined_res, f)


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Reassemble the refined patches into json file.')
    parser.add_argument('dt_json',
                        help='path to coarse masks (json format)')
    parser.add_argument('gt_json',
                        help='path to annotations (json format)')
    parser.add_argument('res_pkl',
                        help='path to network\'s output (pkl format).')
    parser.add_argument('details_dir',
                        help='path to detail_dir.')
    parser.add_argument('out_json',
                        help='where to save the output refined masks.',
                        default='refined.json')
    parser.add_argument('--padding',
                        help='padding (half width)',
                        default=0, type=int)
    parser.add_argument('--num-proc',
                        help='num of process',
                        type=int, default=20)
    args = parser.parse_args()

    start()
