import PIL.Image as Image
import os

original_dir = '../../datasets/coco_stuff/val_img'
target_dir = './datasets/coco164k/images/val2017_256'
os.makedirs(target_dir, exist_ok=True)

img_list = os.listdir(original_dir)
for img in img_list:
    orig_img_name = os.path.join(original_dir, img)
    target_img_name = os.path.join(target_dir, img)

    original_img = Image.open(orig_img_name)
    new_img = original_img.resize((256, 256), resample=Image.NEAREST)
    new_img.save(target_img_name)