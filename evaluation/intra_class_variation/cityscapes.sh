export code=compute_intra_class_variation.py
export date=1105
export device=oem
export expr_name="$date"_cVASIS_learnAll_all_"$device"
export dataset_mode='cityscapes'

export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/"$dataset_mode"/"$expr_name"/test_best/images/synthesized_image
python $code --img_path $img_path --save_name spade_variation