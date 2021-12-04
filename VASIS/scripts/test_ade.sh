#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=1
export batchSize=1
export epoch=best
export date=1105
export device=oem
export ckpt="./checkpoints/ade20k"
export result="./results/ade20k"
export name="$date"_spade_"$device"

export name=CLADE_ICPE

python test.py --name $name \
--norm_mode clade --batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--dataset_mode ade20k \
--results_dir "$result" --checkpoints_dir $ckpt \
--dataroot "./../datasets/ADEChallengeData2016_back" \
--pad 'zero' \
--pos 'learn_relative' --pos_nc 'all' --add_dist --dist_type 'offline' \
--noise_nc 'all'




# compute FID
export real_path=./../datasets/ADEChallengeData2016_back/
export fake_path=./results/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name ade20k


#export real_path=./../datasets/cityscapes/
#export fake_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/
#python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name cityscapes