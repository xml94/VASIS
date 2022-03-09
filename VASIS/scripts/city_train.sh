##############
#  Cityscapes
#############
export gpu=0,1,2,3
export batch=16
export freq=10
export niter=100
export niter_decay=100
export device=oem
export date=2201
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"

##########################################################################
## Ours method with SPADE or CLADE: semantic noise and position code
##########################################################################
export norm_mode=clade_variation

export batch=8
python train.py --name "$date"_"$norm_mode"_norm_avg_learn_all \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
--check_flop 1


#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_avg_fix_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_avg' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1

#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_fix_learn_relative_all \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--check_flop 1


#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_avg_all_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_avg' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_random_cat_all_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'random_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_zero_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'zero' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_all \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_one_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'one' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_fix_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_fix_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_fix_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_fix_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1
#
#export batch=8
#python train.py --name "$date"_"$norm_mode"_norm_cat_all_no \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'no' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1


############################################
## spade ablation study
############################################
# spade original with zero padding
#python train.py --name "$date"_spade_zero_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
## spade reflect padding
#python train.py --name "$date"_spade_reflect_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'


############################################
## clade ablation study
############################################
# clade original with zero padding
#python train.py --name "$date"_clade_zero_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--pos 'learn_relative' --pos_nc 'all' --dist_type 'offline' \
#--noise_nc 'all'

## clade original with reflect padding
#python train.py --name "$date"_clade_reflect_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --dist_type 'offline' \
#--noise_nc 'all'
#
## clade ICPE
#python train.py --name "$date"_clade_ICPE_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
#############################################
### Ours method with CLADE: semantic noise and position code
#############################################
## position: learn, relative, all, noise: all
#python train.py --name "$date"_cVASIS_learnRelativeAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cVASIS_learnRelativeAll_no_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' \
#--noise_nc 'zero'
#
#python train.py --name "$date"_cVASIS_no_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'no' --pos_nc 'all' \
#--noise_nc 'all'
#
#####################################
#python train.py --name "$date"_cVASIS_learnRelativeOne_one_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cVASIS_learnAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cVASIS_relativeAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'relative' --pos_nc 'all' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cVASIS_fixAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'fix' --pos_nc 'all' \
#--noise_nc 'all'

