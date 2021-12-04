from os.path import join
import shutil
import os


source_path = '/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/gtFine_256/val'
target_path = '/home/oem/Mingle/SemanticImageSynthesis/datasets/cityscapes/seg/'
os.makedirs(target_path)


# for root, dnames, fnames in sorted(os.walk(dir)):
#     for fname in fnames:
#         if is_image_file(fname):
#             path = os.path.join(root, fname)
#             images.append(path)

for root, dnames, fnames in sorted(os.walk(source_path)):
    for fname in fnames:
        if '_labelIds' in fname:
            source_name = join(source_path, root, fname)
            target_name = join(target_path, fname)
            shutil.copyfile(source_name, target_name)