#python train.py --name va_oasis_cityscapes_seed100 --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
#--dataroot ./../datasets/cityscapes --batch_size 16 --no_3dnoise --seed 100

#python train.py --name va_oasis_cityscapes_avg --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
#--dataroot ./../datasets/cityscapes --batch_size 8 --no_3dnoise --seed 10

#python train.py --name va_oasis_cityscapes_avg_noseed --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
#--dataroot ./../datasets/cityscapes --batch_size 12 --no_3dnoise --seed 0

python train.py --name va_oasis_cityscapes_cat_seed19_batch16 --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
--dataroot ./../datasets/cityscapes --batch_size 16 --no_3dnoise --seed 19

python train.py --name va_oasis_cityscapes_cat_seed0_batch16 --dataset_mode cityscapes --gpu_ids 0,1,2,3 \
--dataroot ./../datasets/cityscapes --batch_size 16 --no_3dnoise --seed 0