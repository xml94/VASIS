export name=va_oasis_ade20k_seed0_batch24_cat
export gpu=0,1,2,3
python test.py --name ${name} --dataset_mode ade20k --gpu_ids ${gpu} \
--dataroot ./../datasets/ADEChallengeData2016 --batch_size 24 \
--no_3dnoise --seed 0


# make the testing results as our format
export tgt_dir="./results/ade20k/${name}/test_best/images/synthesized_image"
mkdir -p ${tgt_dir}
cp "./results/${name}/best/image/"* ${tgt_dir}

# compute FID
export real_path="./../datasets/ADEChallengeData2016/images_256/validation/"
export fake_path=${tgt_dir}
python fid_score.py $real_path $fake_path --batch-size 1 --gpu ${gpu}