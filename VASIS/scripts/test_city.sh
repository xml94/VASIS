export gpu=1
export batchSize=1
export epoch='best'
export date=2204
export result="results/cityscapes"
export ckpt="./checkpoints/cityscapes"


export kernel_norm=3
export norm_mode=clade_variation
export name="2208_clade_variation_norm_cat_no"
#export name="$date"_"$norm_mode"_kernel_"$kernel_norm"_norm_cat_all_relative_all

# ours
python test.py --name $name \
--norm_mode "$norm_mode" --batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--results_dir "$result" --checkpoints_dir $ckpt \
--dataset_mode cityscapes \
--dataroot ./../datasets/cityscapes \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'no' --pos_nc 'all' --add_dist --dist_type 'offline' \
--kernel_norm $kernel_norm --check_flop 1


#export norm_mode=clade
#python test.py --name $name \
#--norm_mode "$norm_mode" --batchSize $batchSize \
#--gpu_ids $gpu --which_epoch $epoch \
#--results_dir "$result" --checkpoints_dir $ckpt \
#--dataset_mode cityscapes \
#--dataroot ./../datasets/cityscapes \
#--pad 'zero' \
#--mode_noise 'norm_cat' --noise_nc 'zero' \
#--pos 'no' --pos_nc 'all' --add_dist --dist_type 'offline' \
#--kernel_norm $kernel_norm --check_flop 1



# compute FID
export real_path="/home/oem/Mingle/VASIS/datasets/cityscapes/leftImg8bit_256/val/"
export fake_path="./${result}/${name}/test_${epoch}/images/synthesized_image"
python fid_score.py $real_path $fake_path --batch-size 1 --gpu ${gpu} --load_np_name cityscapes
python fid_score.py $real_path $fake_path --batch-size 1 --gpu ${gpu}
