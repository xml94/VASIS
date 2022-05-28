export gpu=0,1,2,3
export freq=5
export batch=32
export niter=150
export niter_decay=0
export device=oem
export date=2204
export result="./checkpoints/coco"
export ckpt="./checkpoints/coco"


############################################
## Ours
############################################
export norm_mode=spade_variation
export batch=28
export niter=150
export niter_decay=0
python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_all \
--dataset_mode coco --dataroot "./../datasets/cocostuff" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode "$norm_mode" \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all'