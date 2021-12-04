import os
from PIL import Image
import argparse
import os.path as osp


# down sample ground truth labels to compute mIoU and Pixel Accuracy
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='/home/oem/Mingle/Project/datasets/cityscapes')
parser = parser.parse_args()

orig_root = osp.join(parser.dataset_dir, 'gtFine/val')
tgt_root = osp.join(parser.dataset_dir, 'gtFine_256/val')

dir_names = os.listdir(orig_root)
for dir_ in dir_names:
    img_names = os.listdir(os.path.join(orig_root, dir_))
    os.makedirs(os.path.join(tgt_root, dir_), exist_ok=True)
    for img in img_names:
        if '.png' in img:
            abs_orig_img = os.path.join(orig_root, dir_, img)
            abs_tgt_img = os.path.join(tgt_root, dir_, img)

            img = Image.open(abs_orig_img)
            img = img.resize((512,256), resample=Image.NEAREST) # sure this, not other method
            img.save(abs_tgt_img)


orig_root = osp.join(parser.dataset_dir, 'gtFine/val')
tgt_root = osp.join(parser.dataset_dir, 'gtFine_256_val')

dir_names = os.listdir(orig_root)
for dir_ in dir_names:
    img_names = os.listdir(os.path.join(orig_root, dir_))
    os.makedirs(os.path.join(tgt_root), exist_ok=True)
    for img in img_names:
        if 'labelIds.png' in img:
            abs_orig_img = os.path.join(orig_root, dir_, img)
            abs_tgt_img = os.path.join(tgt_root, img)

            img = Image.open(abs_orig_img)
            img = img.resize((512,256), resample=Image.NEAREST) # sure this, not other method
            img.save(abs_tgt_img)

# to compute FID with trainging dataset
# orig_root = osp.join(parser.dataset_dir, 'gtFine/train')
# tgt_root = osp.join(parser.dataset_dir, 'gtFine_256/train')
#
# dir_names = os.listdir(orig_root)
# for dir_ in dir_names:
#     img_names = os.listdir(os.path.join(orig_root, dir_))
#     os.makedirs(os.path.join(tgt_root, dir_), exist_ok=True)
#     for img in img_names:
#         if '.png' in img:
#             abs_orig_img = os.path.join(orig_root, dir_, img)
#             abs_tgt_img = os.path.join(tgt_root, dir_, img)
#
#             img = Image.open(abs_orig_img)
#             img = img.resize((512,256), resample=Image.NEAREST) # sure this, not other method
#             img.save(abs_tgt_img)