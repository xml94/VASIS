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


############################################
## clade ablation study
############################################
# clade original with zero padding
python train.py --name "$date"_clade_zero_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'zero' \
--pos 'learn_relative' --pos_nc 'all' --dist_type 'offline' \
--noise_nc 'all'

# clade original with reflect padding
python train.py --name "$date"_clade_reflect_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --dist_type 'offline' \
--noise_nc 'all'

# clade ICPE
python train.py --name "$date"_clade_ICPE_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq "$freq" --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'zero' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

############################################
## Ours method with CLADE: semantic noise and position code
############################################
# position: learn, relative, all, noise: all
python train.py --name "$date"_cVASIS_learnRelativeAll_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

python train.py --name "$date"_cVASIS_learnRelativeAll_no_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' \
--noise_nc 'zero'

python train.py --name "$date"_cVASIS_no_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'no' --pos_nc 'all' \
--noise_nc 'all'

####################################
python train.py --name "$date"_cVASIS_learnRelativeOne_one_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

python train.py --name "$date"_cVASIS_learnAll_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn' --pos_nc 'all' \
--noise_nc 'all'

python train.py --name "$date"_cVASIS_relativeAll_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'relative' --pos_nc 'all' \
--noise_nc 'all'

python train.py --name "$date"_cVASIS_fixAll_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'fix' --pos_nc 'all' \
--noise_nc 'all'


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
## Ours method with SPADE: semantic noise and position code
############################################
#python train.py --name "$date"_sVASIS_learnRelativeAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
## position: learn, All, noise: all
#python train.py --name "$date"_cityscapes_spade_variation_reflect_learnAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'all'
#
## position: no, noise: all
#python train.py --name "$date"_cityscapes_spade_variation_reflect_no_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'no' --pos_nc 'all' \
#--noise_nc 'all'
#
## position: learn, relative, all, noise: no
#python train.py --name "$date"_cityscapes_spade_variation_reflect_learnRelativeAll_no_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq --results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'zero'


##############
#  ADE20K
#############
export batch=32


#python train.py --name "date"ade20kcladevariationreflectlearnAllall′′device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--norm_mode clade_variation --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'all'

#python train.py --name "date"ade20kspadevariation′′device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--norm_mode spade_variation --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' --pos 'fix' --noise_nc 'one'
#
#python train.py --name "date"ade20kspade′′device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--norm_mode spade --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' --pos 'fix' --noise_nc 'one'
#
#python train.py --name "date"ade20kcladevariation′′device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--norm_mode clade_variation --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' --pos 'fix' --noise_nc 'one'
#
#python train.py --name "date"ade20kclade′′device" \
#--dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
#--norm_mode clade --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' --pos 'fix' --noise_nc 'one'


##############
#  COCO-stuff
#############