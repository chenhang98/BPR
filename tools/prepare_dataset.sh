coarse_dir_train=$1
coarse_dir_val=$2
prefix=$3

IOU_THRESH=${IOU_THRESH:-0.25}
IMG_DIR_TRAIN=${IMG_DIR_TRAIN:-'data/cityscapes/leftImg8bit/train'}
IMG_DIR_VAL=${IMG_DIR_VAL:-'data/cityscapes/leftImg8bit/val'}
GT_JSON_TRAIN=${GT_JSON:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json'}
GT_JSON_VAL=${GT_JSON:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'}
BPR_ROOT=${BPR_ROOT:-'.'}

coarse_json_train=${prefix}/coarse_train.json
coarse_json_train_filtered=${prefix}/coarse_train_filtered.json
coarse_json_val=${prefix}/coarse_val.json
dataset_dir=${prefix}/patches


set -x
GREEN='\033[0;32m'
END='\033[0m\n'

mkdir $prefix


printf ${GREEN}"build training set ..."${END}
# convert to json
python $BPR_ROOT/tools/cityscapes2json.py \
    $coarse_dir_train \
    $GT_JSON_TRAIN \
    $coarse_json_train

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
    --mode train 


printf ${GREEN}"build validation set ..."${END}
# convert to json
python $BPR_ROOT/tools/cityscapes2json.py \
    $coarse_dir_val \
    $GT_JSON_VAL \
    $coarse_json_val

# split to patches
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json_val \
    $GT_JSON_VAL \
    $IMG_DIR_VAL \
    $dataset_dir \
    --iou-thresh $IOU_THRESH \
    --mode val 