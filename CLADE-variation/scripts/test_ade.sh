#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=1
export batchSize=1
export epoch=best
export date=1001
export device=oem
export name=ade20k_dist


python test.py --name $name \
--norm_mode clade --batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--dataset_mode ade20k \
--dataroot "./../datasets/ADEChallengeData2016" \
--pad 'zero' --pos 'abs_learn' --noise_nc 'one' \
--add_dist --dist_type 'online'
#--vis 1


# compute FID
export real_path=./../datasets/cityscapes/
export fake_path=./results/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name ade20k


#export real_path=./../datasets/cityscapes/
#export fake_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes