##############
#  Cityscapes
#############
export gpu=0,1,2,3
export batch=16
export freq=10
export niter=100
export niter_decay=100
export device=oem
export date=2204
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"

##########################################################################
## Ours method with SPADE or CLADE: semantic noise and position code
##########################################################################
export norm_mode=spade_variation


export batch=8
python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_one_backup \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
--check_flop 1