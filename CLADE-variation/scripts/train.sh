##############
#  Cityscapes
#############
export gpu=0,1,2,3
export freq=1
export batch=16
export niter=100
export niter_decay=100
export device=oem
export date=1101

# position: learn, relative, all, noise: all
python train.py --name "$date"_cityscapes_clade_variation_reflect_learnRelativeAll_all_"$device" \
--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
--train_eval --eval_epoch_freq $freq \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'

#python train.py --name "$date"_cityscapes_spade_variation_reflect_learnRelativeAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--noise_nc 'all'
#
## position: learn, All, noise: all
#python train.py --name "$date"_cityscapes_spade_variation_reflect_learnAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cityscapes_clade_variation_reflect_learnAll_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'all'
#
## position: no, noise: all
#python train.py --name "$date"_cityscapes_spade_variation_reflect_no_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'no' --pos_nc 'all' \
#--noise_nc 'all'
#
#python train.py --name "$date"_cityscapes_clade_variation_reflect_no_all_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'no' --pos_nc 'all' \
#--noise_nc 'all'
#
## position: learn, relative, all, noise: no
#python train.py --name "$date"_cityscapes_spade_variation_reflect_learnRelativeAll_no_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode spade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'zero'
#
#python train.py --name "$date"_cityscapes_clade_variation_reflect_learnRelativeAll_no_"$device" \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes_back" \
#--norm_mode clade_variation --gpu_ids $gpu --batchSize $batch --niter $niter --niter_decay $niter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'reflect' \
#--pos 'learn' --pos_nc 'all' \
#--noise_nc 'zero'


#export batch=8
#export gpu=0
#horovodrun -np 4 -H 113.198.60.51:4 -p 12345 python train.py --name test \
#--dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
#--norm_mode clade --gpu_ids gpu−−batchSizebatch --niter niter−−niterdecayniter_decay \
#--train_eval --eval_epoch_freq $freq \
#--pad 'zero' --pos 'abs_learn' --noise_nc 'all' \
#--pos_nc 'all' \
#--env 'horovod'

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