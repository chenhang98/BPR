config=$1
ckpt=$2
coarse_json=$3
prefix=$4

IOU_THRESH=${IOU_THRESH:-0.25}
IMG_DIR=${IMG_DIR:-'data/coco/val2017'}
GT_JSON=${GT_JSON:-'data/coco/annotations/instances_val2017.json'}
BPR_ROOT=${BPR_ROOT:-'.'}
GPUS=${GPUS:-4}

out_pkl=${prefix}/refined.pkl
out_json=${prefix}/refined.json
dataset_dir=${prefix}/patches


set -x
GREEN='\033[0;32m'
END='\033[0m\n'

mkdir $prefix


printf ${GREEN}"build patches dataset ..."${END}
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json \
    $GT_JSON \
    $IMG_DIR \
    $dataset_dir \
    --iou-thresh $IOU_THRESH

printf ${GREEN}"inference the network ..."${END}
DATA_ROOT=$dataset_dir \
bash $BPR_ROOT/tools/dist_test_float.sh \
    $config \
    $ckpt \
    $GPUS \
    --out $out_pkl

printf ${GREEN}"reassemble ..."${END}
python $BPR_ROOT/tools/merge_patches.py \
    $coarse_json \
    $GT_JSON \
    $out_pkl \
    $dataset_dir/detail_dir/val \
    $out_json