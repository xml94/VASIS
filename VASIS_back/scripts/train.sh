##############
#  Cityscapes
#############
export gpu=0,1,2,3
export freq=10
export batch=8
export niter=100
export niter_decay=100
export device=oem
export date=1105
export result="./results/cityscapes"
export ckpt="./checkpoints/cityscapes"

python train.py --name "$date"_cVASIS_learnRelativeAll_all_semanticNoiseInput_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all' \
--input_type 'noise'


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
export result="./results/ade20k"
export ckpt="./checkpoints/ade20k"


python train.py --name "$date"_cVASIS_learnRelativeAll_all_semanticNoiseInput_"$device" \
--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016_back" \
--gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" \
--results_dir "$result" --checkpoints_dir "$ckpt" \
--norm_mode clade_variation \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all' \
--input_type 'noise'