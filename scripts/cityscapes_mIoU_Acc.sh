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


export model="CLADE-variation"
export epoch="best"
export gpu=2
export batch_size=1
export date=1001
export device=oem
export expr_name=1030_cityscapes_clade_variation_reflect_learnRelativeAll_all_oem

sh ./evaluation/drn-master/eval_city.sh $expr_name $model $epoch $gpu $batch_size
