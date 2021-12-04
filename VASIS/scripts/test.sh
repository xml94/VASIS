#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=1
export batchSize=1
export epoch=best
export date=1030
export device=oem
export name=1030_cityscapes_clade_variation_reflect_learnRelativeAll_all_oem

python test.py --name $name \
--norm_mode clade_variation --batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--dataset_mode cityscapes \
--dataroot ./../datasets/cityscapes_back \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'


# compute FID
export real_path=./../datasets/cityscapes/
export fake_path=./results/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes


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