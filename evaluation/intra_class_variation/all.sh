export code=compute_intra_class_variation.py
#
#python $code
#
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/spade_cityscapes/test_latest/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name spade_original
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/0727_cityscapes_spade_oem/test_best/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name spade_retrain
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE_SPADE_ReduceIntraVariation/results/0826_cityscapes_spade_reduceIntraVariation_withoutZeroPad_oem/test_best/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name spade_reflect
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE_SPADE_ReduceIntraVariation/results/0826_cityscapes_spade_reduceIntraVariation_withoutZeroPad_allConv1x1_oem/test_best/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name spade_allConv1x1
#
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/clade_cityscapes/test_best/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name clade_original
#export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/0901_cityscapes_clade_zero_oem/test_best/images/synthesized_image
#python $code --img_path $img_path \
#  --save_name clade_retrain
export img_path=/home/oem/Mingle/SemanticImageSynthesis/CLADE-variation/results/1030_cityscapes_clade_variation_reflect_learnRelativeAll_all_oem/test_best/images/synthesized_image
python $code --img_path $img_path \
  --save_name spade_variation
