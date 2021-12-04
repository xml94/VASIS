export gpu=3
export batchSize=10
export epoch=best
export date=0926
export device=oem
export name="$date"_cityscapes_spade_variation_"$device"
export model=CLADE-variation

export real_path=./datasets/cityscapes/gtFine_256_val
export fake_path=./$model/results/$name/test_"$epoch"/images/synthesized_image
python ./scripts/density_coverage.py $real_path $fake_path --batch-size $batchSize --gpu $gpu #--load_np_name cityscapes
