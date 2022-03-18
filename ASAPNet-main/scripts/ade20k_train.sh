##############
#  ADE20K
#############
export gpu=0,1,2,3
export freq=5
export device=oem
export date=2203
export result="./results/ade20k"
export ckpt="./checkpoints/ade20k"

export batch=4
export niter=150
export niter_decay=150
python train.py --name "$date"_FVASIS \
  --dataset_mode ade20k --dataroot "./../datasets/ADEChallengeData2016" \
  --gpu_ids $gpu \
  --batchSize $batch --niter $niter --niter_decay $niter_decay \
  --train_eval --eval_epoch_freq $freq \
  --results_dir "$result" --checkpoints_dir $ckpt \
  --height 256 --width 256 \
  --add_dist --dist_type 'offline'