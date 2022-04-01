##############
#  Cityscapes
#############
export gpu=0,1,2,3
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
export norm_mode=spade_variation
export kernel_norm=3
export batch=8
python train.py --name "$date"_"$norm_mode"_norm_cat_all_learn_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' \
--kernel_norm $kernel_norm



export norm_mode=spade_variation
export kernel_norm=1
export batch=8


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_avg_learn_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_random_cat_all_learn_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'random_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' \
--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_no_learn_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'zero' \
--pos 'learn' --pos_nc 'one' \
--kernel_norm $kernel_norm


#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_one_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'one' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_learn_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm


#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_learn_all \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#

#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_fix_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_fix_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_fix_learn_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_fix_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm
#
#
#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_learn_relative_one \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm


#python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_no \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode "$norm_mode" --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'no' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_avg_all_fix_learn_relative_one \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'fix_learn_relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_avg_all_fix_learn_relative_all \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_cat_all_fix_learn_relative_all \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm


python train.py --name "$date"_"$norm_mode"_kernel_1_norm_avg_all_learn_all \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm