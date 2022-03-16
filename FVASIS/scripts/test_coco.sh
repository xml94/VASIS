#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=1
export batchSize=1
export epoch=best
export device=oem
export date=2201
export result="results/coco"
export ckpt="checkpoints/coco"


#export norm_mode=spade_variation
#export name="$date"_"$norm_mode"_norm_cat_all_learn_one
#python test.py --name $name \
#--norm_mode spade_variation --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--dataset_mode coco --dataroot "./../datasets/cocostuff" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1


#export norm_mode=clade_variation
#export name="$date"_"$norm_mode"_norm_avg_all_fix_learn_relative_all
#python test.py --name $name \
#--norm_mode clade_variation --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--dataset_mode coco --dataroot "./../datasets/cocostuff" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'reflect' \
#--mode_noise 'norm_avg' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--check_flop 1


#export norm_mode=spade
#export name=spade_coco_pretrained
#python test.py --name $name \
#--norm_mode spade_variation --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--dataset_mode coco --dataroot "./../datasets/cocostuff" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--mode_noise 'norm_cat' --noise_nc 'zero' \
#--pos 'no' --pos_nc 'one' --add_dist --dist_type 'offline' \
#--check_flop 1

#export norm_mode=clade
#export name=coco
#python test.py --name $name \
#--norm_mode clade --batchSize $batchSize \
#--gpu_ids 2 --which_epoch $epoch \
#--dataset_mode coco --dataroot "./../datasets/cocostuff" \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--pad 'zero' \
#--mode_noise 'norm_cat' --noise_nc 'zero' \
#--pos 'no' --pos_nc 'one' --dist_type 'offline' \
#--check_flop 1

export norm_mode=clade
export name=coco_dist
python test.py --name $name \
--norm_mode clade --batchSize $batchSize \
--gpu_ids 3 --which_epoch $epoch \
--dataset_mode coco --dataroot "./../datasets/cocostuff" \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'zero' \
--mode_noise 'norm_cat' --noise_nc 'zero' \
--pos 'relative' --pos_nc 'one' --add_dist --dist_type 'offline' \
--check_flop 1

# compute FID
export real_path=./../datasets/cityscapes/
export fake_path=./$result/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name coco


#export real_path=./../datasets/cityscapes/
#export fake_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes