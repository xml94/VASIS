#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=1
export batchSize=1
export epoch=best
export device=oem
export date=1105
export result="results/coco"
export ckpt="checkpoints/coco"
export name="$date"_sVASIS_learnRelativeAll_all_epoch150_batch28_"$device"


python test.py --name $name \
--norm_mode spade_variation --batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--dataset_mode coco --dataroot "./../datasets/cocostuff" \
--results_dir "$result" --checkpoints_dir $ckpt \
--pad 'reflect' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'online' \
--noise_nc 'all'
#--vis 1


# compute FID
export real_path=./../datasets/cityscapes/
export fake_path=./$result/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name coco


#export real_path=./../datasets/cityscapes/
#export fake_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes