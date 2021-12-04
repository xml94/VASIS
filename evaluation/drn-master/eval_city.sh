#!/usr/bin/env sh
expr_name=$1
model=$2
epoch=$3
GPU=$4
batch=$5


# must use absolute directory
workDir=$(pwd)
export drn_data_dir="$workDir"/datasets/eval_cityscapes/cityscapes

export syn_dir=./$model/results/cityscapes/$expr_name/test_"$epoch"/images/synthesized_image
export real_dir='./datasets/cityscapes'
export drn_model_dir='./pretrained_model/drn-d-105_ms_cityscapes.pth'


# copy synthesized data into the folder to be checked by drn-master
if [ ! -e $drn_data_dir ]
then
  mkdir -p "$drn_data_dir"
fi
# make sure that every time predict the generated images
if [ -e "$drn_data_dir"/leftImg8bit ]
then
  rm -rf "$drn_data_dir"/leftImg8bit
fi
cp -r $syn_dir "$drn_data_dir"/leftImg8bit

# resizing original real images
export gtFine_tgt_dir="$real_dir"/gtFine_256
if [ ! -e $gtFine_tgt_dir ]
then
  echo 'Checking: resizing image to size 256...'
  python ./evaluation/drn-master/resize_cityscapes.py --dataset_dir $real_dir
else
  echo 'Checking: doing nothing since size 256 images exist.'
fi

# generate required files: val_labels (to compute mIoU) and val_images
# and make sure that you already have info.json
rm -rf "$drn_data_dir/gtFine"
cp -r $gtFine_tgt_dir "$drn_data_dir/gtFine"
python ./evaluation/drn-master/datasets/cityscapes/prepare_data.py "$drn_data_dir/gtFine"
find "$drn_data_dir/leftImg8bit" -maxdepth 3 -name "*_leftImg8bit.png" | sort > "$drn_data_dir/val_images.txt"
find "$drn_data_dir/gtFine/val" -maxdepth 3 -name "*_trainIds.png" | sort > "$drn_data_dir/val_labels.txt"

# compute mIoU and Pixel Accuracy
if [ ! -e $drn_model_dir ]
then
  echo "Error: There is no pretrained model."
fi
CUDA_VISIBLE_DEVICES=$GPU python3 ./evaluation/drn-master/segment.py test \
-d $drn_data_dir \
--arch drn_d_105 \
-c 19 \
--pretrained "$drn_model_dir" \
--batch-size $batch \
--with-gt \
--phase val