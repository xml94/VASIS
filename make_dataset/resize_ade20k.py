# resize image to 256, 256 to display
import os
from PIL import Image
import argparse
import os.path as osp


# down sample ground truth labels to compute mIoU and Pixel Accuracy
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='./../datasets/ADEChallengeData2016')
parser = parser.parse_args()

# images
orig_root = osp.join(parser.dataset_dir, 'images/validation')
tgt_root = osp.join(parser.dataset_dir, 'images_256/validation')
os.makedirs(os.path.join(tgt_root), exist_ok=True)
dir_names = os.listdir(orig_root)
img_names = sorted(os.listdir(orig_root))
for img in img_names:
    if '.jpg' in img:
        abs_orig_img = os.path.join(orig_root, img)
        abs_tgt_img = os.path.join(tgt_root, img)

        img = Image.open(abs_orig_img)
        img = img.resize((256, 256), resample=Image.BICUBIC)
        img.save(abs_tgt_img)
