##############
#  ADE20K
#############
export gpu=0,1,2,3
export freq=10
export batch=16
export niter=100
export niter_decay=100
export device=oem
export date=1105
export result="./results/outdoor_ade20k"
export ckpt="./checkpoints/outdoor_ade20k"


############################################
## Ours
############################################
python train.py --name "$date"_cVASIS_learnRelativeAll_all_average_"$device" \
--dataset_mode ade20k --dataroot "./../datasets/ADE20K_Outdoor" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode clade_variation \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

python train.py --name "$date"_sVASIS_learnRelativeAll_all_average_"$device" \
--dataset_mode ade20k --dataroot "./../datasets/ADE20K_Outdoor" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode spade_variation \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'