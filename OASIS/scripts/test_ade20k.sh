export name=oasis_ade20k_pretrained
python test.py --name ${name} --dataset_mode ade20k --gpu_ids 0,1,2,3 \
--dataroot ./../datasets/ADEChallengeData2016 --batch_size 4 \
--seed 100


# make the testing results as our format
export tgt_dir="./results/ade20k/${name}/test_best/images/synthesized_image"
mkdir -p ${tgt_dir}
cp "./results/${name}/best/image/"* ${tgt_dir}

# compute FID
export gpu=0
export real_path="./../datasets/ADEChallengeData2016/images_256/validation/"
export fake_path=${tgt_dir}
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu