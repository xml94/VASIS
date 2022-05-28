export gpu=1
export batchSize=10
export epoch='best'
export date=2204
export result="results/cityscapes"
export ckpt="./checkpoints/cityscapes"
export norm_mode=spade_variation
export kernel_norm=1
#export name="$date"_"$norm_mode"_kernel_"$kernel_norm"_norm_cat_all_relative_all
#export name="2204_spade_variation_kernel_3_norm_cat_all_fix_learn_relative_all"
export name="Cityscapes_AreaNorm_2Linear_2Conv3x3_Conv1x1"

#python test.py --name $name \
#--norm_mode "$norm_mode" --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--dataset_mode cityscapes \
#--dataroot ./../datasets/cityscapes \
#--pad 'reflect' \
#--mode_noise 'norm_cat' --noise_nc 'all' \
#--pos 'fix_learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm --check_flop 1

#python test.py --name $name \
#--norm_mode "$norm_mode" --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--dataset_mode cityscapes \
#--dataroot ./../datasets/cityscapes \
#--pad 'reflect'


# compute FID
export real_path="/home/oem/Mingle/VASIS/datasets/cityscapes/leftImg8bit_256/val/"
export fake_path="./${result}/${name}/test_${epoch}/images/synthesized_image"
#export fake_path="/home/oem/Mingle/VASIS/VASIS/results/cityscapes/spade_cityscapes/test_${epoch}/images/synthesized_image"
#export fake_path="/home/oem/Mingle/VASIS/VASIS/results/cityscapes/clade_cityscapes/test_${epoch}/images/synthesized_image"
#export fake_path="/home/oem/Mingle/VASIS/VASIS/results/cityscapes/clade_cityscapes_dist/test_${epoch}/images/synthesized_image"
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu


#python test.py --name checkpoints/cityscapes_dist \
#--norm_mode clade --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--dataset_mode cityscapes \
#--dataroot "./../datasets/cityscapes" \
#--pad 'zero' --pos 'learn' --noise_nc 'one' \
#--add_dist \
#--check_time 1


#python test.py --name checkpoints/ade20k_dist \
#--norm_mode clade --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--dataset_mode ade20k \
#--dataroot "./../datasets/ADEChallengeData2016" \
#--pad 'zero' --pos 'learn' --noise_nc 'one' \
#--add_dist \
#--check_time 1
#
#
## compute FID
#export real_path=./../datasets/cityscapes/
#export fake_path=./results/checkpoints/ade20k_dist/test_"$epoch"/images/synthesized_image
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name ade20k