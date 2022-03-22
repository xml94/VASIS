#python test.py --name [model_name] --norm_mode clade --batchSize 1 --gpu_ids 0 --which_epoch best --dataset_mode [dataset] --dataroot [Path_to_dataset] --add_dist

export gpu=3
export batchSize=1
export epoch='latest'
export date=2201
export result="results/cityscapes"
export ckpt="./checkpoints/cityscapes"
export norm_mode=spade_variation
export name=ASAPNet_VASIS

python test.py --name $name \
  --dataset_mode cityscapes --dataroot "./../datasets/cityscapes" \
  --gpu_ids $gpu \
  --batchSize $batchSize \
  --nThreads 8


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