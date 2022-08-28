##############
#  Cityscapes
#############
export gpu=0,1,2,3
export freq=10
export niter=100
export niter_decay=100
export device=oem
export date=2208
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"
export kernel_norm=3
export batch=16


# retrain SPADE
#python train.py --name "$date"_spade_retrain \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode spade --gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--mode_noise 'norm_cat' --noise_nc 'zero' \
#--pos 'no' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm



##########################################################################
## Ours method with SPADE or CLADE: semantic noise and position code
##########################################################################
export norm_mode=spade_variation

# no_no
python train.py --name "$date"_"$norm_mode"_no_no \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'zero' \
--pos 'no' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# norm_cat_no
python train.py --name "$date"_"$norm_mode"_norm_cat_no \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'no' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# no_learn
python train.py --name "$date"_"$norm_mode"_no_learn \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'zero' \
--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# norm_cat_learn
python train.py --name "$date"_"$norm_mode"_norm_cat_learn \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# norm_avg_learn
python train.py --name "$date"_"$norm_mode"_norm_avg_learn \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_avg' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# random_cat_learn
python train.py --name "$date"_"$norm_mode"_random_cat_learn \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'random_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'all' \
--kernel_norm $kernel_norm

## norm_cat_fix
python train.py --name "$date"_"$norm_mode"_norm_cat_fix \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'fix' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# norm_cat_relative
python train.py --name "$date"_"$norm_mode"_norm_cat_relative \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm

# norm_cat_fix_learn_relative
python train.py --name "$date"_"$norm_mode"_norm_cat_fix_learn_relative \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--norm_mode "$norm_mode" --gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm
