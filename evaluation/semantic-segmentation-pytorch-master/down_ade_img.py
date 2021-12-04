import PIL.Image as Image
import os

orig_root = './../../datasets/ADEChallengeData2016/images/validation'
target_root = 'data/ADEChallengeData2016/FID_256/'

if not os.path.isdir(target_root):
    os.mkdir(target_root)

names_img = os.listdir(orig_root)
for name in names_img:
    abs_orig_name = os.path.join(orig_root, name)
    abs_new_name = os.path.join(target_root, name)

    img = Image.open(abs_orig_name)
    img = img.resize((256, 256), resample=Image.NEAREST)
    img.save(abs_new_name)