export name="va_oasis_cityscapes"
python test.py --name ${name} --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
--dataroot ./../datasets/cityscapes --batch_size 20 --no_3dnoise \
--seed 41

# make the testing results as our format
export tgt_dir="./results/cityscapes/${name}/test_best/images/synthesized_image"
mkdir -p ${tgt_dir}
cp "./results/${name}/best/image/"* ${tgt_dir}


# compute FID
export gpu=0
export real_path="./../datasets/cityscapes/leftImg8bit_256/val/"
export fake_path=${tgt_dir}
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu
