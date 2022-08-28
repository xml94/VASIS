# resize image to 512, 256 to display
import os
from PIL import Image
import argparse
import os.path as osp


# down sample ground truth labels to compute mIoU and Pixel Accuracy
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./../datasets/cityscapes')
parser = parser.parse_args()

# images
orig_root = osp.join(parser.dataset_dir, 'leftImg8bit/val')
tgt_root = osp.join(parser.dataset_dir, 'leftImg8bit_256/val')
os.makedirs(tgt_root, exist_ok=True)
dir_names = os.listdir(orig_root)
for dir_ in dir_names:
    img_names = os.listdir(os.path.join(orig_root, dir_))
    # os.makedirs(os.path.join(tgt_root, dir_), exist_ok=True)
    for img in img_names:
        if '.png' in img:
            abs_orig_img = os.path.join(orig_root, dir_, img)
            abs_tgt_img = os.path.join(tgt_root, img)

            img = Image.open(abs_orig_img)
            img = img.resize((512, 256), resample=Image.BICUBIC)
            img.save(abs_tgt_img)

# labels
orig_root = osp.join(parser.dataset_dir, 'gtFine/val')
tgt_root = osp.join(parser.dataset_dir, 'gtFine_256/val_color')
dir_names = os.listdir(orig_root)
for dir_ in dir_names:
    img_names = os.listdir(os.path.join(orig_root, dir_))
    os.makedirs(os.path.join(tgt_root), exist_ok=True)
    for img in img_names:
        if 'color.png' in img:
            abs_orig_img = os.path.join(orig_root, dir_, img)
            abs_tgt_img = os.path.join(tgt_root, img)

            img = Image.open(abs_orig_img)
            img = img.resize((512, 256), resample=Image.NEAREST)
            img.save(abs_tgt_img)
