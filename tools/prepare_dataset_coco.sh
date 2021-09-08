coarse_json_train=$1
coarse_json_val=$2
prefix=$3
n_samples=${4:-'-1'}

IOU_THRESH=${IOU_THRESH:-0.25}
IMG_DIR_TRAIN=${IMG_DIR_TRAIN:-'data/coco/train2017'}
IMG_DIR_VAL=${IMG_DIR_VAL:-'data/coco/val2017'}
GT_JSON_TRAIN=${GT_JSON:-'data/coco/annotations/instances_train2017.json'}
GT_JSON_VAL=${GT_JSON:-'data/coco/annotations/instances_val2017.json'}
BPR_ROOT=${BPR_ROOT:-'.'}

coarse_json_train_filtered=${prefix}/coarse_train_filtered.json
dataset_dir=${prefix}/patches


set -x
GREEN='\033[0;32m'
END='\033[0m\n'

mkdir $prefix


printf ${GREEN}"build training set ..."${END}

# filter IoU < 0.5
python $BPR_ROOT/tools/filter.py \
    $coarse_json_train \
    $GT_JSON_TRAIN \
    $coarse_json_train_filtered

# split to patches
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json_train_filtered \
    $GT_JSON_TRAIN \
    $IMG_DIR_TRAIN \
    $dataset_dir \
    --iou-thresh $IOU_THRESH \
    --mode train \
    --sample-inst $n_samples


printf ${GREEN}"build validation set ..."${END}

# split to patches
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json_val \
    $GT_JSON_VAL \
    $IMG_DIR_VAL \
    $dataset_dir \
    --iou-thresh $IOU_THRESH \
    --mode val 