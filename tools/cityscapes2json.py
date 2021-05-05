import os
import os.path as osp
import cv2
import json
import numpy as np
from tqdm import *
from collections import defaultdict
from pycocotools import mask as maskUtils
from fire import Fire


class Cityscape:
    def __init__(self, gt_file, anno_dir):
        self.anno_dir = anno_dir
        self.gt_file = gt_file
        self._img2id, self._id2img = self._build_gt()
        self._data = []
        self._id2annos = self._build()

    def _load_anno(self, anno_file):
        mask = cv2.imread(anno_file, 0)
        mask[mask > 0] = 1
        ys, xs = np.where(mask)
        if not len(xs):
            return None, None, None
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        box = [x1, y1, x2-x1+1, y2-y1+1]
        submask = mask[y1:y2+1, x1:x2+1]
        area = submask.astype(np.int).sum()
        seg = maskUtils.encode(np.asfortranarray(mask))
        return box, area, seg

    def _build_gt(self):
        img2id, id2img = dict(), dict()
        with open(self.gt_file) as f:
            imgs = json.load(f)['images']
            for im in imgs:
                img_name = im['file_name'].split(
                    '/')[1].split('_leftImg8bit')[0]
                img2id[img_name] = im['id']
                id2img[im['id']] = im['file_name']
        return img2id, id2img

    def _build(self):
        all_files = os.listdir(self.anno_dir)
        all_files = [_ for _ in all_files if _.endswith('.txt')]
        print("Loading annotations")

        # img_id -> anno_file, category_id, score
        id_to_anno_info = defaultdict(list)
        cnt = 0
        for af in tqdm(all_files):
            img_name = af.split('_leftImg8bit_pred')[0]
            img_name = img_name.split('_index.txt')[0]  # for segfix
            img_id = self._img2id[img_name]
            with open(osp.join(self.anno_dir, af)) as f:
                for line in f.readlines():
                    fn, cid, score = line.strip().split()
                    item = dict(anno_file=fn, category_id=int(cid),
                                score=float(score))
                    item['iscrowd'] = 0
                    item['id'] = cnt
                    item['bbox'], _, item['segmentation'] = \
                        self._load_anno(osp.join(self.anno_dir, fn))
                    item['image_id'] = img_id
                    cnt += 1
                    if item['bbox'] is None:
                        continue

                    item['bbox'] = [int(_) for _ in item['bbox']]
                    item['segmentation']['counts'] = \
                        item['segmentation']['counts'].decode('utf8')
                    self._data.append(item)
        print("Done")
        return id_to_anno_info


def main(dt_root, gt_json, out_json):
    cs = Cityscape(gt_json, dt_root)
    with open(out_json, 'w') as f:
        json.dump(cs._data, f)


if __name__ == '__main__':
    Fire(main)
