model=$1
expr_name=$2
epoch=$3
gpu=$4


export syn_dir=./$model/results/$expr_name/test_"$epoch"/images/synthesized_image
export eval_dir=./datasets/eval_cocostuff/datasets/coco164k
export model_dir=./pretrained_model/deeplabv2_resnet101_msc-cocostuff164k-100000.pth
export coco_dir=./datasets/coco_stuff

# copy synthesized images
if [ ! -e "$eval_dir/images/val2017" ]
then
  mkdir -p "$eval_dir/images/val2017"
fi
rm -rf "$eval_dir/images/val2017"
cp -r "$syn_dir/" "$eval_dir/images/val2017/"

# copy ground truth to evaluate
if [ ! -e "$eval_dir/annotations/val2017" ]
then
  mkdir -p "$eval_dir/annotations/val2017"
  rm -rf "$eval_dir/annotations/val2017"
  cp -r "$coco_dir/val_label" "$eval_dir/annotations/val2017"
fi

# please check ./evaluation/deeplab_pytorch/configs/cocostuff164k.yaml
CUDA_VISIBLE_DEVICES=$gpu python ./evaluation/deeplab_pytorch/main.py test \
    --config-path ./evaluation/deeplab_pytorch/configs/cocostuff164k.yaml \
    --model-path $model_dir \
    --cuda