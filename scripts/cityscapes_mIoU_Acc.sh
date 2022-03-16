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


export model="VASIS"
export epoch="best"
export gpu=2
export batch_size=1
export date=2201
export device=oem
export norm_mode=spade_variation
export name="$date"_"$norm_mode"_kernel_1_norm_avg_all_fix_learn_relative_all

sh ./evaluation/drn-master/eval_city.sh $name $model $epoch $gpu $batch_size
