export name=oasis_coco_pretrained
#python test.py --name oasis_coco_pretrained --dataset_mode coco --gpu_ids 0,1,2,3 \
#--dataroot ./../datasets/cocostuff --batch_size 8


# make the testing results as our format
export tgt_dir="./results/coco/${name}/test_best/images/synthesized_image"
mkdir -p ${tgt_dir}
cp "./results/${name}/best/image/"* ${tgt_dir}

# compute FID
export gpu=0
export real_path="./../datasets/cocostuff/val_img_256"
export fake_path=${tgt_dir}
python fid_score.py $real_path $fake_path --batch-size 1 --gpu $gpu