export gpu=0,1,2,3
export freq=5
export batch=32
export niter=100
export niter_decay=0
export device=oem
export date=2201
export result="./checkpoints/coco"
export ckpt="./checkpoints/coco"


############################################
## Ours
############################################
export batch=28
python train.py --name "$date"_sVASIS_learnRelativeAll_all_"$device" \
--dataset_mode coco --dataroot "./../datasets/cocostuff" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode spade_variation \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

export batch=32
python train.py --name "$date"_cVASIS_learnRelativeAll_all_"$device" \
--dataset_mode coco --dataroot "./../datasets/cocostuff" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode clade_variation \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'