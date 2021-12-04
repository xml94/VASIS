import os
import PIL.Image as Image

orig_root = '../../../../datasets/cityscapes/leftImg8bit/val'
tgt_root = './gtFine_FID'
dir_names = os.listdir(orig_root)
city_names = os.listdir(orig_root)
for city_name in city_names:
    img_names = os.listdir(os.path.join(orig_root, city_name))
    for img_name in img_names:
        original = os.path.join(orig_root, city_name, img_name)
        target = os.path.join(tgt_root, img_name)

        img = Image.open(original)
        img = img.resize((512, 256), resample=Image.NEAREST)
        img.save(target)