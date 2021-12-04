export code=compute_intra_class_variation.py
export date=1105
export device=oem
export seg_path='/home/oem/Mingle/SemanticImageSynthesis/datasets/ADEChallengeData2016_back/annotations/validation'

# real img
export real_img_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/ADEChallengeData2016/images/validation_256/
#python $code --img_path $real_img_path --seg_path $seg_path --save_name real --data_type 'ade20k'

# fake img
expr_name="$date"_spade_"$device"
export expr_name=CLADE
export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/"$expr_name"/test_best/images/synthesized_image
python $code --img_path $img_path --seg_path $seg_path --save_name cladeICPE --data_type 'ade20k'


# one img for fake and real
#expr_name="$date"_clade_zero_"$device"
#export one_img_name="ADE_val_00000001.png"
#export img_path="/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/$expr_name/test_best/images/fake"
#mkdir -p $img_path
#rm $img_path/*
#cp "/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/$expr_name/test_best/images/synthesized_image/$one_img_name" "$img_path"
#python $code --img_path $img_path --seg_path $seg_path --save_name spade_variation --data_type 'ade20k'
#
#export real_img_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/ADEChallengeData2016/images/validation_256
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/datasets/ADEChallengeData2016/images/validation_256_one
#mkdir -p $img_path
#export one_img_name="ADE_val_00000001.jpg"
#rm $img_path/*
#cp $real_img_path/$one_img_name $img_path
#python $code --img_path $img_path --seg_path $seg_path --save_name spade_variation --data_type 'ade20k'
