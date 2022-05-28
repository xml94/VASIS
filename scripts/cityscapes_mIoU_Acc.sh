:<<EOF
This shell aims to compute mIoU and Acc for cityscapes dataset.
expr_name: experiment name, you should have same name directory in which
  there is trained model
model: model name, same with model's directory name
  <SPADE-master>, <CLADE-main>, <CLADE_Padding>, ...
epoch: which epoch you want to check, for example, best, latest, 100
  <best>, <latest>, <100>, <130>
gpu: which gpu you want to use, 0, 0,1,2
  <2>, <0,1>
batch_size: how many batches when computing
EOF


export model="OASIS"
export epoch="best"
export gpu=1
export batch_size=5
export date=2204
export device=oem
export norm_mode=spade_variation
export kernel_norm=3
#export name="$date"_"$norm_mode"_kernel_"$kernel_norm"_norm_cat_all_relative_all
export name="oasis_cityscapes_pretrained"

sh ./evaluation/drn-master/eval_city.sh $name $model $epoch $gpu $batch_size


#export model="ASAPNet-main"
#export epoch="latest"
#export gpu=2
#export batch_size=5
#export date=2201
#export device=oem
#export norm_mode=spade_variation
#export name=
#
#sh ./evaluation/drn-master/eval_city.sh $name $model $epoch $gpu $batch_size
