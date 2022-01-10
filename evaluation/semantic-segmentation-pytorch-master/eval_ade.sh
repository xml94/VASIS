export model=$1
export expr_name=$2
export epoch=$3
export gpu=$4


export syn_dir=./$model/results/ade20k/$expr_name/test_"$epoch"/images/synthesized_image
export eval_dir="./datasets/eval_ade20k/data/ADEChallengeData2016"
export ade20k_dir='./datasets/ADEChallengeData2016/annotations'

export basic_dir="./evaluation/semantic-segmentation-pytorch-master"
export model_dir="./../../pretrained_model/ade20k-resnet101-upernet"
export ade_eval_dir="./../..datasets/eval_ade20k/data"

mkdir -p "$eval_dir"

# copy synthesized images to be evaluate
if [ ! -e "$eval_dir/images/validation" ]
then
  mkdir -p "$eval_dir/images/validation"
fi
rm -rf "$eval_dir/images/validation"
cp -r $syn_dir "$eval_dir/images/validation"

# copy ground truth
if [ ! -e "$eval_dir/annotations/validation" ]
then
  mkdir -p "$eval_dir/annotations/validation"
  cp -r "$ade20k_dir/validation" "$eval_dir/annotations"
fi

# please make dataset directory correct by
# rewriting the config/ade20k-resnet101-upernet.yaml
cd $basic_dir

python3 eval_multipro.py \
--gpus $gpu \
--cfg ./config/ade20k-resnet101-upernet.yaml

cd ../..