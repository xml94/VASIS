##############
#  Cityscapes
#############
export gpu=0,1,2,3
export batch=16
export freq=1
export niter=100
export niter_decay=100
export device=oem
export date=2203
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"


#export batch=8
#python train.py --name "$date"_FVASIS__ \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--gpu_ids $gpu \
#--batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--height 256 --width 512 \
#--add_dist --dist_type 'offline' \
#--check_flop 1 \
#--netG FVASIS --continue_train

export batch=8
python train.py --name "$date"_transformer \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
--gpu_ids $gpu \
--batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--results_dir "$result" --checkpoints_dir $ckpt \
--height 256 --width 512 \
--add_dist --dist_type 'offline' \
--check_flop 1 \
--netG FVASISTRANS