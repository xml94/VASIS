#python test.py --name oasis_cityscapes_pretrained --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
#--dataroot ./../datasets/cityscapes --batch_size 20


# compute FID
export name="oasis_cityscapes_pretrained"
export gpu=0
export real_path="./../datasets/cityscapes/leftImg8bit_256/val/"
export fake_path="./results/${name}/best/image"
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu