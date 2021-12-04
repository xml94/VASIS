from os.path import join
import shutil
import os
from PIL import Image
from tqdm import tqdm


source_path = '/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/leftImg8bit/val'
target_path = '/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/'
os.makedirs(target_path, exist_ok=True)

osize = (512, 256)

for root, dnames, fnames in sorted(os.walk(source_path)):
    for fname in tqdm(fnames):
        source_name = join(source_path, root, fname)
        target_name = join(target_path, fname)
        # shutil.copyfile(source_name, target_name)

        source_img = Image.open(source_name).convert('RGB')
        target_img = source_img.resize(osize)
        target_img.save(target_name)
