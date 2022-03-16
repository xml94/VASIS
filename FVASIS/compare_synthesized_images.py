"""
This code aims to make us compare several images visually in one website.
Sometimes we use several method or hyper-parameters to do one experiments,
and latter we want to compare these different results.
Unfortunately, each experiment save its results in different directory,
which make us inconvenient to compare them, especially when more than
two comparisons.
This code aims to put all images in different setting with same name
into one website, and then it is efficient to compare them.
Input:
    dir1/prefix:
        img1.jpg
        img2.jpg
        ...
    dir2/prefix:
        img1.jpg
        img2.jpg
        ...
    ...
"""

import my_html as html
import argparse
import os
import os.path as osp
import shutil


parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default="compared",
                    help='the directory to save html page')
parser.add_argument('--prefix', type=str, default="test_best/images/synthesized_image",
                    help='the prefix for each directory: test_latest/images/synthesized_image')
parser.add_argument('--dataset', type=str, default='cocostuff',
                    help='cityscapes, ade20k, cocostuff')
parser = parser.parse_args()

save_dir = parser.save_dir

# make the compared dir
if parser.dataset == 'cityscapes':
    compare_dir = "./results/cityscapes"
    save_dir = save_dir + '_cityscapes'
    ours = "2201_spade_variation_norm_cat_all_learn_one"
    parser.dirs = [
        f"{compare_dir}/spade_cityscapes",
        f"{compare_dir}/{ours}",
        f"{compare_dir}/clade_cityscapes",
        f"{compare_dir}/clade_cityscapes_dist",
        f"{compare_dir}/2201_clade_variation_norm_avg_all_fix_learn_relative_all",
    ]
    source_img_dir = "./../datasets/cityscapes/leftImg8bit_256/val/"
    source_seg_dir = f"{compare_dir}/{ours}/test_best/images/input_label/"
elif parser.dataset == 'ade20k':
    compare_dir = "./results/ade20k"
    save_dir = save_dir + '_ade20k'
    ours = "2201_spade_variation_norm_cat_all_learn_one"
    parser.dirs = [
        f"{compare_dir}/spade_ade20k",
        f"{compare_dir}/{ours}",
        f"{compare_dir}/clade_ade20k",
        f"{compare_dir}/clade_dist_ade20k",
        f"{compare_dir}/2201_clade_variation_norm_avg_all_fix_learn_relative_all",
    ]
    source_img_dir = "./../datasets/ADEChallengeData2016/images_256/validation/"
    source_seg_dir = f"{compare_dir}/{ours}/test_best/images/input_label/"
elif parser.dataset == 'cocostuff':
    compare_dir = "./results/coco"
    save_dir = save_dir + '_coco'
    ours = "2201_spade_variation_norm_cat_all_learn_one"
    parser.dirs = [
        f"{compare_dir}/spade_coco_pretrained",
        f"{compare_dir}/{ours}",
        f"{compare_dir}/coco",
        f"{compare_dir}/coco_dist",
        f"{compare_dir}/2201_clade_variation_norm_avg_all_fix_learn_relative_all",
    ]
    source_img_dir = "./../datasets/cocostuff/val_img_256/"
    source_seg_dir = f"{compare_dir}/{ours}/test_best/images/input_label/"
else:
    print('Check the dataset mode please...')


# prepare directory if with prefix
dirs = []
if parser.prefix is not None:
    for i in range(len(parser.dirs)):
        dirs.append(osp.join(parser.dirs[i], parser.prefix))

# prepare image names
correct_names = []
filenames = os.listdir(dirs[0])
for name in filenames:
    correct_names.append(name)

webpage = html.HTML(save_dir, 'Comparison', refresh=0)
for name in sorted(correct_names):
    webpage.add_header(f'{name}')
    ims, txts, links = [], [], []

    # copy original image
    new_img_name = osp.join(save_dir, 'images', 'img', name)
    os.makedirs(osp.join(save_dir, 'images', 'img'), exist_ok=True)
    if parser.dataset == 'cityscapes':
        old_img_name = osp.join(source_img_dir, name)
    elif parser.dataset == 'ade20k':
        old_img_name = osp.join(source_img_dir, name.replace('.png', '.jpg'))
    elif parser.dataset == 'cocostuff':
        old_img_name = osp.join(source_img_dir, name.replace('.png', '.jpg'))
    else:
        pass
    shutil.copyfile(old_img_name, new_img_name)
    new_img_name = osp.join('img', name)
    ims.append(new_img_name)
    txts.append('gt')
    links.append(new_img_name)

    # copy semantic segmentation
    new_img_name = osp.join(save_dir, 'images', 'seg', name)
    os.makedirs(osp.join(save_dir, 'images', 'seg'), exist_ok=True)
    old_img_name = osp.join(source_seg_dir, name)
    shutil.copyfile(old_img_name, new_img_name)
    new_img_name = osp.join('seg', name)
    ims.append(new_img_name)
    txts.append('seg')
    links.append(new_img_name)

    # copy translated image
    for i in range(len(dirs)):
        new_img_name = osp.join(save_dir, 'images', osp.basename(parser.dirs[i]) + '_' + name)
        os.makedirs(osp.join(save_dir, 'images', osp.basename(parser.dirs[i])), exist_ok=True)
        old_img_name = osp.join(dirs[i], name)
        shutil.copyfile(old_img_name, new_img_name)
        new_img_name = osp.join(osp.basename(parser.dirs[i]) + '_' + name)
        ims.append(new_img_name)
        txts.append(osp.basename(parser.dirs[i]))
        links.append(new_img_name)
    webpage.add_images(ims, txts, links, width=400)

webpage.save()