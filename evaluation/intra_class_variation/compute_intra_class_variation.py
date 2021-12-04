"""
compute the statistics for every label of image
input:
    -semantic segmentation
    -image
output
    -mean for each label
    -std for each label

Be careful
    3 channels
    many semantic labels
    many images

"""
import os

import numpy as np
from os.path import join
import argparse
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms


def get_transform(osize=(256, 256)):
    transform_list = []
    transform_list.append(transforms.Resize(osize, interpolation=Image.NEAREST))
    transform_list += [transforms.ToTensor()]
    return transforms.Compose(transform_list)


parser = argparse.ArgumentParser()
parser.add_argument('--seg_path', type=str, default='/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/seg/')
parser.add_argument('--img_path', type=str, default='/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/img/')
parser.add_argument('--data_type', type=str, default='cityscapes', choices=['cityscapes', 'ade20k', 'cocostuff'])
parser.add_argument('--save_name', type=str, default='cityscapes_val')
parser.add_argument('--type_variation', type=str, default='one_color',
                    choices=['one_color', 'all_color', 'one_light', 'all_light'])
parser = parser.parse_args()


RGB2GRAY = torch.tensor((0.299, 0.587, 0.114)).view(1, 3)

seg_path = parser.seg_path
img_path = parser.img_path

if parser.data_type == 'cityscapes':
    label_nc = 35
    osize = (256, 512)
elif parser.data_type == 'ade20k':
    label_nc = 150
    osize = (256, 256)
else:
    assert parser.data_type == 'cocostuff'
    label_nc = 183
    osize = (256, 256)


img_name_list = sorted(os.listdir((img_path)))
num_img = len(img_name_list)


one_color_label = torch.zeros((label_nc, num_img))
one_light_label = torch.zeros((label_nc, num_img))
all_color_label = [[] for j in range(label_nc)]  # [label_nc, 1, n]
all_light_label = [[] for j in range(label_nc)]  # [label_nc, 1, n]
num = 0
for img_name in tqdm(img_name_list):
    if parser.data_type == 'cityscapes':
        seg_name = img_name.replace('leftImg8bit', 'gtFine_labelIds')
    else:
        seg_name = img_name.replace('.jpg', '.png')

    seg = Image.open(join(seg_path, seg_name))
    transform_label = get_transform(osize=osize)
    seg = transform_label(seg) * 255
    seg[seg == 255] = label_nc

    _, h, w = seg.size()
    input_label = torch.FloatTensor(label_nc, h, w).zero_()
    seg_one_hot = input_label.scatter_(0, seg.long(), 1.0)

    img = plt.imread(join(img_path, img_name))
    img = torch.FloatTensor(img) * 255
    img = img.permute(2, 0, 1)

    for s in range(label_nc):
        mask_area = torch.sum(seg_one_hot.bool()[s])
        if mask_area > 1:
            img_label = img.masked_select(seg_one_hot.bool()[s]).reshape(3, mask_area)
            one_color_label[s, num] = img_label.std(dim=1).mean()
            one_light_label[s, num] = torch.std(torch.mm(RGB2GRAY, img_label))
            all_color_label[s].append(img_label.numpy())
            all_light_label[s].append(torch.mm(RGB2GRAY, img_label).numpy())
    num += 1

one_color = one_color_label.mean(dim=1).numpy()
one_light = one_light_label.mean(dim=1).numpy()
all_color, all_light = np.zeros((label_nc)), np.zeros((label_nc))
for s in range(label_nc):
    if len(all_color_label[s]) > 0:
        all_color[s] = np.hstack(all_color_label[s]).std(axis=1).mean()
    if len(all_light_label[s]) > 0:
        all_light[s] = np.hstack(all_light_label[s]).std(axis=1).mean()

os.makedirs(parser.save_name, exist_ok=True)
dataset = np.array((one_color.mean(), one_light.mean(),
                    all_color.mean(), all_light.mean()))
dataset_name = join(parser.save_name, 'dataset.txt')
np.savetxt(dataset_name, dataset, fmt='%.3f', delimiter=',', newline=',')

print(f'checking...: {parser.save_name}: one_color: {one_color.mean()},'
      f'one_light: {one_light.mean()}',
      f'all_color: {all_color.mean()}',
      f'all_light: {all_light.mean()}',)


one_color_name = join(parser.save_name, 'one_color.txt')
one_light_name = join(parser.save_name, 'one_light.txt')
all_color_name = join(parser.save_name, 'all_color.txt')
all_light_name = join(parser.save_name, 'all_light.txt')

np.savetxt(one_color_name, one_color, fmt='%.3f', delimiter=',', newline=',')
np.savetxt(one_light_name, one_light, fmt='%.3f', delimiter=',', newline=',')
np.savetxt(all_color_name, all_color, fmt='%.3f', delimiter=',', newline=',')
np.savetxt(all_light_name, all_light, fmt='%.3f', delimiter=',', newline=',')
