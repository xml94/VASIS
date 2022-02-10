export gpu=1
export batchSize=16
export epoch=best
export date=2201
export device=oem
export ckpt="./checkpoints/ade20k"
export result="./results/ade20k"

export norm_mode=spade_variation

export name="$date"_"$norm_mode"_norm_cat_all_learn_one_epoch300_batch32
python test.py --name $name \
--batchSize $batchSize \
--gpu_ids $gpu --which_epoch $epoch \
--dataset_mode ade20k \
--results_dir "$result" --checkpoints_dir $ckpt \
--dataroot "./../datasets/ADEChallengeData2016" \
--norm_mode "$norm_mode" \
--pad 'reflect' \
--mode_noise 'norm_cat' --noise_nc 'all' \
--pos 'learn' --pos_nc 'one' --add_dist --dist_type 'offline' \

# compute FID
export real_path=./../datasets/ADEChallengeData2016/
export fake_path=./$result/$name/test_"$epoch"/images/synthesized_image
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu --load_np_name ade20k
