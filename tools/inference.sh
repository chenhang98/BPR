config=$1
ckpt=$2
coarse_dir=$3
prefix=$4

IOU_THRESH=${IOU_THRESH:-0.55}
IMG_DIR=${IMG_DIR:-'data/cityscapes/leftImg8bit/val'}
GT_JSON=${GT_JSON:-'data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json'}
BPR_ROOT=${BPR_ROOT:-'.'}
GPUS=${GPUS:-4}

coarse_json=${prefix}/coarse.json
out_pkl=${prefix}/refined.pkl
out_json=${prefix}/refined.json
out_dir=${prefix}/refined
dataset_dir=${prefix}/patches


set -x

mkdir $prefix

echo "convert to json format ..."
python $BPR_ROOT/tools/cityscapes2json.py \
    $coarse_dir \
    $GT_JSON \
    $coarse_json

echo "build patches dataset ..."
python $BPR_ROOT/tools/split_patches.py \
    $coarse_json \
    $GT_JSON \
    $IMG_DIR \
    $dataset_dir \
    --iou-thresh $IOU_THRESH

echo "inference the network ..."
DATA_ROOT=$dataset_dir \
bash $BPR_ROOT/tools/dist_test_float.sh \
    $config \
    $ckpt \
    $GPUS \
    --out $out_pkl

echo "reassemble ..."
python $BPR_ROOT/tools/merge_patches.py \
    $coarse_json \
    $GT_JSON \
    $out_pkl \
    $dataset_dir/detail_dir/val \
    $out_json

echo "convert to cityscape format ..."
python $BPR_ROOT/tools/json2cityscapes.py \
    $out_json \
    $GT_JSON \
    $out_dir