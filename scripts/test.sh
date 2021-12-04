:<<EOF
This shell aims to generate fake images and compute FID.
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

export expr_name="clade_cityscapes_semanticNoise_channel1_multiply"
export model="CLADE_Padding"
export dataset=""
export data_mode=""
export epoch="latest"
export gpu=1
export batch_size=1

#sh ./$expr_name/test.py