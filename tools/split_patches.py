import numpy as np
import json
import cv2
import os
import os.path as osp
from tqdm import tqdm
from argparse import ArgumentParser
from pycocotools.coco import COCO

import torch
import torch.nn.functional as F
import warnings
import multiprocessing as mp

try:
    from mmdet.ops.nms import nms
    ops = 'mmdet'
except Exception:
    from mmcv.ops.nms import nms
    ops = 'mmcv'


coco = None


def cal_iou(mask1, mask2):
    si = np.sum(mask1 & mask2)
    su = np.sum(mask1 | mask2)
    return si / su


def query_gt_mask(maskdt, coco, imgid, catid):
    # search the gt mask with max IoU
    annids = coco.getAnnIds(imgIds=imgid, catIds=catid)
    anns = coco.loadAnns(annids)

    masks = []
    for ann in anns:
        if not ann['iscrowd']:
            masks.append(coco.annToMask(ann))
    ious = [cal_iou(maskdt, _) for _ in masks]

    if len(ious) > 0:
        ind = ious.index(max(ious))
        return masks[ind]
    else:
        return np.zeros(maskdt.shape)


def find_float_boundary(maskdt, width=3):
    # Extract boundary from instance mask
    maskdt = torch.Tensor(maskdt).unsqueeze(0).unsqueeze(0)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt.permute(1, 0, 2, 3), boundary_finder,
                             stride=1, padding=width//2).permute(1, 0, 2, 3)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask[0, 0].numpy()


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
    sdets = sdets.copy()
    s = sdets[:, 0] < 0
    sdets[s, 0] = 0
    sdets[s, 2] = patch_size

    s = sdets[:, 1] < 0
    sdets[s, 1] = 0
    sdets[s, 3] = patch_size

    s = sdets[:, 2] >= W
    sdets[s, 0] = W - 1 - patch_size
    sdets[s, 2] = W - 1

    s = sdets[:, 3] >= H
    sdets[s, 1] = H - 1 - patch_size
    sdets[s, 3] = H - 1
    return sdets


def get_dets(maskdt, patch_size, iou_thresh=0.3):
    """Generate patch proposals from the coarse mask.

    Args:
        maskdt (array): H,W
        patch_size (int): [description]
        iou_thresh (float, optional): useful for nms. Defaults to 0.3.

    Returns:
        array: filtered bboxs. shape N, 4. each row contain x1, y1, 
            x2, y2, score. e.g.
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
    """
    fbmask = find_float_boundary(maskdt)
    ys, xs = np.where(fbmask)
    scores = fbmask[ys, xs]
    dets = np.stack([xs-patch_size//2, ys-patch_size//2,
                     xs+patch_size//2, ys+patch_size//2, scores]).T
    if ops == 'mmdet':
        _, inds = nms(dets, iou_thresh)
    else:
        _, inds = nms(np.ascontiguousarray(dets[:, :4], np.float32),
                      np.ascontiguousarray(dets[:, 4], np.float32),
                      iou_thresh)
    sdets = dets[inds]

    H, W = maskdt.shape
    return _force_move_back(sdets, H, W, patch_size)


def save_details(inst_id, dets, image_name, category, patches, root_dir):
    """save to details dir, format: e.g.
        {'image_name': 'frankfurt/frankfurt_000001_005410_leftImg8bit.png', 
        'category': {'id': 24, 'name': 'person'}, 
        'patches': [0], 
        'hlist': [(y, h)],
        'wlist': [(x, w)]}

    Args:
        inst_id (int): instance's id
        dets (array): each row: x1, y1, x2, y2, score
        image_name (str): e.g. frankfurt/frankfurt_000001_005410_leftImg8bit.png
        category (dict): e.g. {'id': 24, 'name': 'person'}
        patches (list[int]): patch's ids
        root_dir (str): dataset root dir
    """
    subroot = osp.join(root_dir, "detail_dir", args.mode, f"{inst_id}.txt")
    dets = dets.astype(int)
    xs, ys = dets[:, 0], dets[:, 1]
    ws = dets[:, 2] - dets[:, 0]
    hs = dets[:, 3] - dets[:, 1]
    sdict = dict(image_name=image_name,
                 category=category,
                 patches=patches,
                 hlist=list(zip(ys.tolist(), hs.tolist())),
                 wlist=list(zip(xs.tolist(), ws.tolist())))
    with open(subroot, "w") as f:
        f.write(str(sdict))


def save_patch(patch_id, img_patch, dt_patch, gt_patch, root_dir):
    """save patches to png files

    Args:
        patch_id (int): an unique id
        img_patch (array): shape h, w, 3
        dt_patch (array): shape h, w
        gt_patch (array): shape h, w
        root_dir (str): dataset root dir
    """
    fn = f"{patch_id}.png"
    cv2.imwrite(
        osp.join(root_dir, "img_dir", args.mode, fn), img_patch)
    cv2.imwrite(
        osp.join(root_dir, "mask_dir", args.mode, fn), dt_patch)
    cv2.imwrite(
        osp.join(root_dir, "ann_dir", args.mode, fn), gt_patch)


def crop(img, maskdt, maskgt, dets, padding):
    # padding
    dets = dets.astype(int)[:, :4] + padding
    pd = padding
    img = np.pad(img, ((pd, pd), (pd, pd), (0, 0)))
    maskdt = np.pad(maskdt, pd)
    maskgt = np.pad(maskgt, pd)

    # crop
    img_patches, dt_patches, gt_patches = [], [], []
    for x1, y1, x2, y2 in dets:
        img_patches.append(img[y1-pd:y2+pd, x1-pd:x2+pd, :])
        dt_patches.append(maskdt[y1-pd:y2+pd, x1-pd:x2+pd])
        gt_patches.append(maskgt[y1-pd:y2+pd, x1-pd:x2+pd])
    return img_patches, dt_patches, gt_patches


def run_inst(i_inst):
    inst_id, inst = i_inst
    global coco

    imgid = inst['image_id']
    catid = inst['category_id']
    maskdt = coco.annToMask(inst)
    imgname = coco.imgs[imgid]["file_name"]

    img = cv2.imread(osp.join(args.imgs_dir, imgname))
    h, w, _ = img.shape
    if h <= args.patch_size or w <= args.patch_size:
        return

    maskgt = query_gt_mask(maskdt, coco, imgid, catid)
    dets = get_dets(maskdt, args.patch_size, args.iou_thresh)
    img_patches, dt_patches, gt_patches = crop(
        img, maskdt, maskgt, dets, args.padding)

    patchids = []
    for i in range(len(dets)):
        if i >= args.max_inst:
            warnings.warn(f"pathid overflow! imgid={imgid}, catid={catid}, "
                          f"inst_id={inst_id}, i={i}, imgname={imgname}")
        patchid = inst_id * args.max_inst + i
        save_patch(patchid, img_patches[i], dt_patches[i]*255,
                   gt_patches[i]*255, args.out_dir)
        patchids.append(patchid)

    save_details(inst_id, dets, imgname, coco.cats[catid], patchids,
                 args.out_dir)


def sample_subset(xs, n_samples):
    inds = np.random.choice(len(xs), n_samples)
    return [xs[i] for i in inds]


def start():
    os.system("mkdir %s" % (osp.join(args.out_dir, "img_dir")))
    os.system("mkdir %s" % (osp.join(args.out_dir, "ann_dir")))
    os.system("mkdir %s" % (osp.join(args.out_dir, "mask_dir")))
    os.system("mkdir %s" % (osp.join(args.out_dir, "detail_dir")))
    os.system("mkdir %s" %
              (osp.join(args.out_dir, "img_dir", args.mode)))
    os.system("mkdir %s" %
              (osp.join(args.out_dir, "ann_dir", args.mode)))
    os.system("mkdir %s" %
              (osp.join(args.out_dir, "mask_dir", args.mode)))
    os.system("mkdir %s" %
              (osp.join(args.out_dir, "detail_dir", args.mode)))

    global coco
    coco = COCO(args.gt_json)
    dt = json.load(open(args.dt_json))

    # sample a subset of all instances
    if args.sample_inst > 0:
        dt = sample_subset(dt, args.sample_inst)

    # build patches dataset with multi-process
    with mp.Pool(processes=args.num_proc) as p:
        with tqdm(total=len(dt)) as pbar:
            for _ in p.imap_unordered(run_inst, enumerate(dt)):
                pbar.update()


if __name__ == "__main__":
    parser = ArgumentParser(
        description='Generate patches dataset from json file.')
    parser.add_argument('dt_json',
                        help='path to coarse masks (json format)')
    parser.add_argument('gt_json',
                        help='path to annotations (json format)')
    parser.add_argument('imgs_dir',
                        help='path to images')
    parser.add_argument('out_dir',
                        help='where to save the output patches dataset',
                        default='./patches')
    parser.add_argument('--mode',
                        help='train or val',
                        default='val')
    parser.add_argument('--iou-thresh',
                        help='IoU threshold for patch proposal nms',
                        default=0.25, type=float)
    parser.add_argument('--patch-size',
                        help='patch size',
                        default=64, type=int)
    parser.add_argument('--padding',
                        help='padding (half width)',
                        default=0, type=int)
    parser.add_argument('--num-proc',
                        help='number of process',
                        type=int, default=20)
    parser.add_argument('--max-inst',
                        help='max instances num per image',
                        default=100000)
    parser.add_argument('--sample-inst',
                        help='How many instances to be sampled, -1 means all',
                        default=-1, type=int)
    args = parser.parse_args()

    np.random.seed(2020)
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    start()
