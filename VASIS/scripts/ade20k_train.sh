##############
#  ADE20K
#############
export gpu=0,1,2,3
export freq=5
#export batch=32
#export niter=100
#export niter_decay=100
export device=oem
export date=2205
export result="./results/ade20k"
export ckpt="./checkpoints/ade20k"


export norm_mode=clade_variation
export batch=32
export niter=150
export niter_decay=150
python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_all_epoch300 \
  --dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
  --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
  --train_eval --eval_epoch_freq "$freq" \
  --results_dir "$result" --checkpoints_dir "$ckpt" \
  --norm_mode "$norm_mode" \
  --pad 'reflect' \
  --mode_noise 'norm_cat' --noise_nc 'all' \
  --pos 'learn' --pos_nc 'all'


############################################
## clade ablation study
############################################
# clade original with zero padding
#python train.py --name "$date"_clade_zero_"$device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016_back" \
#--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--norm_mode clade \
#--pad 'zero' \
#--pos 'learn_relative' --pos_nc 'all' --dist_type 'offline' \
#--noise_nc 'all'
##
### clade ICPE
#python train.py --name "$date"_cladeICPE_"$device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016_back" \
#--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--norm_mode clade \
#--pad 'zero' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'


############################################
## SPADE
############################################
#python train.py --name "$date"_spade_epoch300_"$device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" \
#--results_dir "$result" --checkpoints_dir "$ckpt" \
#--norm_mode spade \
#--pad 'zero' \
#--niter 150 --niter_decay 150
