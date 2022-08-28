##############
#  Cityscapes
#############
export gpu=1
export batch=1
export freq=10
export niter=100
export niter_decay=100
export device=oem
export date=2204
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"

export norm_mode=clade_variation
export kernel_norm=3
python train.py --name "$date"_"$norm_mode"_test \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'random_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' \
--kernel_norm $kernel_norm \
--check_flop 1


##############
#  ADE20K
#############
#export gpu=1
#export freq=5
#export device=oem
#export date=2205
#export result="./results/ade20k"
#export ckpt="./checkpoints/ade20k"
#
#
#export norm_mode=spade_variation
#export batch=1
#export niter=150
#export niter_decay=150
#python train.py --name "$date"_"$norm_mode"_temp \
#  --dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#  --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#  --train_eval --eval_epoch_freq "$freq" \
#  --results_dir "$result" --checkpoints_dir "$ckpt" \
#  --norm_mode "$norm_mode" \
#  --pad 'reflect' \
#  --mode_noise 'norm_cat' --noise_nc 'all' \
#  --pos 'learn' --pos_nc 'all' \
#  --check_flop 1


##############
#  COCO
#############
#export gpu=0
#export freq=5
#export batch=1
#export niter=150
#export niter_decay=0
#export device=oem
#export date=2204
#export result="./checkpoints/coco"
#export ckpt="./checkpoints/coco"
#
#
#export norm_mode=spade_variation
#export batch=28
#export niter=150
#export niter_decay=0
#python train.py --name "$date"_"$norm_mode"_temp \
#--dataset_mode coco --dataroot "./../datasets/cocostuff" \
#--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" \
#--results_dir "$result" --checkpoints_dir "$ckpt" \
#--norm_mode "$norm_mode" \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'all' \
#--check_flop 1