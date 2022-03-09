# Please firstly make a conda environment for coco with following command
# conda env create -f evaluation/deeplab_pytorch/configs/conda_env.yaml
# conda activate deeplab_pytorch

# after downloading the dataset
# unzip dataset
#unzip ./../datasets/cocostuff/annotations_trainval2017.zip \
#  -d ./../datasets/cocostuff/
#unzip ./../datasets/cocostuff/stuffthingmaps_trainval2017.zip \
#  -d ./../datasets/cocostuff/
#mv ./../datasets/cocostuff/train2017 ./../datasets/cocostuff/train_label
#mv ./../datasets/cocostuff/val2017 ./../datasets/cocostuff/val_label

# make instance map
cp coco_generate_instance_map.py ./../datasets/cocostuff/
mkdir val_inst
mkdir train_inst
python ./../datasets/cocostuff/coco_generate_instance_map.py \
  --annotation_file './annotations/instances_train2017.json' \
  --input_label_dir './train_label' --output_instance_dir 'train_inst'
python ./../datasets/cocostuff/coco_generate_instance_map.py \
  --annotation_file './annotations/instances_val2017.json' \
  --input_label_dir './val_label' --output_instance_dir 'val_inst'

# make relative distance map
python ./../VASIS/util/cal_dist_masks.py --path ./../datasets/cocostuff --dataset coco