import os 
import os.path as osp
import torch 
import torch.nn.functional as F 
import numpy as np
import cv2
import mmcv
from mmcv.ops.nms import nms
from mmcv.ops.roi_align import roi_align
from tqdm import tqdm
from functools import partial
from torch.utils.data import Dataset, DataLoader

from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel, DataContainer, collate



def find_float_boundary(maskdt, width):
    """Find the boundaries.

    Args:
        maskdt (tensor): shape N, H, W
        width (int): boundary width.

    Returns:
        tensor: shape N, H, W
    """
    N, H, W = maskdt.shape
    maskdt = maskdt.view(N, 1, H, W)
    boundary_finder = maskdt.new_ones((1, 1, width, width))
    boundary_mask = F.conv2d(maskdt, boundary_finder, 
                    stride=1, padding=width//2)
    bml = torch.abs(boundary_mask - width*width)
    bms = torch.abs(boundary_mask)
    fbmask = torch.min(bml, bms) / (width*width/2)
    return fbmask.view(N, H, W)


def _force_move_back(sdets, H, W, patch_size):
    # force the out of range patches to move back
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


def get_dets(fbmask, patch_size, iou_thresh=0.3):
    """boundaries of coarse mask -> patch bboxs

    Args:
        fbmask (tensor): H,W, float boundary mask
        patch_size (int): [description]
        iou_thresh (float, optional): useful for nms. Defaults to 0.3.

    Returns:
        tensor: filtered bboxs. x1, y1, x2, y2, score
    """
    ys, xs = torch.nonzero(fbmask, as_tuple=True)
    scores = fbmask[ys,xs]
    ys = ys.float()
    xs = xs.float()
    dets = torch.stack([xs-patch_size//2, ys-patch_size//2, 
            xs+patch_size//2, ys+patch_size//2, scores]).T
    _, inds = nms(dets[:,:4].contiguous(), 
        dets[:,4].contiguous(), iou_thresh)
    sdets = dets[inds]

    H, W = fbmask.shape
    return _force_move_back(sdets, H, W, patch_size)


class PatchDataset(Dataset):
    def __init__(self, img_paths, dt_paths, device, out_size=(128,128)):
        self.device = device
        self.out_size = out_size
        self.img_mean = np.array([123.675, 116.28, 103.53]).reshape(1,1,3)
        self.img_std = np.array([58.395, 57.12, 57.375]).reshape(1,1,3)
        self._img2dts = list(zip(img_paths, dt_paths))      # list of (img_path, list of coarse_mask_path)

    def __len__(self):
        return len(self._img2dts)

    def __getitem__(self, i):
        img_path, dt_paths = self._img2dts[i]
        img = cv2.imread(img_path)[:,:,::-1]     # BGR -> RGB
        img = np.ascontiguousarray(img)
        img = (img - self.img_mean) / self.img_std

        valid_dt_paths, valid_maskdt = [], []   # skip empty mask
        for dt_path in dt_paths:
            m = cv2.imread(dt_path, 0) > 0
            if m.any():
                valid_dt_paths.append(dt_path)
                valid_maskdt.append(m)
        if len(valid_dt_paths):
            valid_maskdt = np.stack(valid_maskdt)
        else:
            valid_maskdt = np.zeros((0, 1024, 2048), dtype=np.float32)

        return DataContainer([
                valid_dt_paths,
                torch.tensor(img, dtype=torch.float), \
                torch.tensor(valid_maskdt, dtype=torch.float)
            ])


def _build_dataloader(img_paths, dt_paths, device):
    dataset = PatchDataset(img_paths, dt_paths, device)
    return DataLoader(dataset, pin_memory=True, collate_fn=collate)


def _build_model(cfg, ckpt, patch_size=64):
    # build the model and load checkpoint
    cfg = mmcv.Config.fromfile(cfg)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    img_meta = [dict(
        ori_shape=(patch_size, patch_size),
        flip=False)]
    model = build_segmentor(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    return partial(model.module.inference, img_meta=img_meta, rescale=False)


def _to_rois(xyxys):
    inds = xyxys.new_zeros((xyxys.size(0), 1))
    return torch.cat([inds, xyxys], dim=1).float().contiguous()


def split(img, maskdts, boundary_width=3, iou_thresh=0.25, patch_size=64, out_size=128):
    # maskdts: N, H, W
    fbmasks = find_float_boundary(maskdts, boundary_width)

    detss = []
    for i in range(fbmasks.size(0)):
        dets = get_dets(fbmasks[i], patch_size, iou_thresh=iou_thresh)[:,:4]
        detss.append(dets)

    all_dets = torch.cat(detss, dim=0)
    img = img.permute(2,0,1).unsqueeze(0).float().contiguous()   # 1,3,H,W
    img_patches = roi_align(img, _to_rois(all_dets), patch_size)

    _detss = [torch.cat([i*_.new_ones((_.size(0), 1)), _], dim=1) for i,_ in enumerate(detss)]
    _detss = torch.cat(_detss)
    dt_patches = roi_align(maskdts[:,None,:,:], _detss, patch_size)

    img_patches = F.interpolate(img_patches, (out_size, out_size), mode='bilinear')
    dt_patches = F.interpolate(dt_patches, (out_size, out_size), mode='nearest')
    return detss, torch.cat([img_patches, 2*dt_patches-1], dim=1)


def merge(maskdts, detss, maskss, patch_size=64):
    # detss: list of dets (Ni,4), x1,y1,x2,y2 format, len K
    # maskdts: (K, H, W)
    # maskss (sum_i Ni, 128, 128)
    out = []

    K, H, W = maskdts.shape
    maskdts = maskdts.bool()
    maskss = F.interpolate(maskss.unsqueeze(0), (patch_size, patch_size), 
            mode='bilinear').squeeze(0)
    dt_refined = torch.zeros_like(maskdts[0], dtype=torch.float32)  # H, W
    dt_count = torch.zeros_like(maskdts[0], dtype=torch.float32)    # H, W

    p = 0
    for k in range(K):
        dets = detss[k]
        dets = dets[:, :4].int()    # Ni, 4
        maskdt = maskdts[k]         # H, W
        q = p + dets.size(0)
        masks = maskss[p:q]         # Ni, 64, 64
        p = q

        dt_refined.zero_()
        dt_count.zero_()
        for i in range(dets.size(0)):
            x1, y1, x2, y2 = dets[i]
            dt_refined[y1:y2, x1:x2] += masks[i]
            dt_count[y1:y2, x1:x2] += 1

        s = dt_count > 0
        dt_refined[s] /= dt_count[s]
        maskdt[s] = dt_refined[s] > 0.5

        out.append(maskdt)
    return out


def inference(cfg, ckpt, img_paths, dt_paths, out_dir, max_ins=32):
    if not osp.exists(out_dir):
        os.makedirs(out_dir)

    model = _build_model(cfg, ckpt)
    dataloader = _build_dataloader(img_paths, dt_paths, 
            device=torch.device('cuda:0'))

    def _inference_one(img, sub_maskdts, sub_dt_paths): # to save GPU memory
        dets, patches = split(img, sub_maskdts)
        masks = model(patches)[:,1,:,:]         # N, 128, 128
        refineds = merge(sub_maskdts, dets, masks)
        for i, dt_path in enumerate(sub_dt_paths):
            cv2.imwrite(
                osp.join(out_dir, osp.basename(dt_path)),
                refineds[i].cpu().numpy().astype(np.uint8) * 255
            )
        return refineds[i].cpu().numpy().astype(np.uint8) * 255

    # inference on each image
    with tqdm(dataloader) as tloader:
        for dc in tloader:
            dt_paths, img, maskdts = dc.data[0][0]
            if len(dt_paths):
                img = img.cuda()             # 3, 1024, 2048
                maskdts = maskdts.cuda()     # N, 1024, 2048

                p = 0
                for sub_maskdts in maskdts.split(max_ins):
                    q = p + sub_maskdts.size(0)
                    sub_dt_paths = dt_paths[p:q]
                    p = q
                    _inference_one(img, sub_maskdts, sub_dt_paths)


if __name__=='__main__':
    cfg = "../configs/bpr/hrnet18s_128.py"
    ckpt = "../ckpts/hrnet18s_128-24055c80.pth"
    img_paths = ['lindau_000000_000019_leftImg8bit.png', ]            # image
    dt_paths = [['lindau_000000_000019_leftImg8bit_15_car.png'], ]    # coarse mask images: 0 for background, >0 for instance
    inference(cfg, ckpt, img_paths, dt_paths, "./refined", max_ins=32)
