:<<EOF
This shell aims to compute mIoU and Acc for ADE20k dataset.
expr_name: experiment name, you should have same name directory in which
  there is trained model
model: model name, same with model's directory name
  <SPADE-master>, <CLADE-main>, <CLADE_Padding>, ...
epoch: which epoch you want to check, for example, best, latest, 100
  <best>, <latest>, <100>, <130>
gpu: which gpu you want to use, 0, 0,1,2
  <2>, <0,1>
batch_size: how many batches when computing, for ade20k, only 1 is valid
EOF


export model="VASIS"
export epoch="best"
export gpu=1
export batch_size=1
export date=1105
export device=oem
export expr_name="$date"_sVASIS_learnRelativeAll_all_batch28_epoch300_oem50_"$device"

sh ./evaluation/semantic-segmentation-pytorch-master/eval_ade.sh $model $expr_name $epoch $gpu