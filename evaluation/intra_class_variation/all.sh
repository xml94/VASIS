export code=compute_intra_class_variation.py

export img_path=/home/oem/Mingle/VASIS/VASIS/results/ade20k/1105_sVASIS_learnRelativeAll_all_batch28_epoch300_oem/test_best/images/synthesized_image
export seg_path=/home/oem/Mingle/VASIS/datasets/ADEChallengeData2016/annotations/validation/
python $code --img_path $img_path --seg_path $seg_path --save_name spade_variation --data_type ade20k
