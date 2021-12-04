""" Downloads and extracts the COCO-Stuff Dataset for use with SPADE """

import argparse
import pathlib
import sys
import urllib.request
import zipfile
import os


cocofiles = {
    "val2017.zip": "http://images.cocodataset.org/zips/",
    #"stuffthingmaps_trainval2017.zip": "http://calvin-vision.net/wp-content/uploads/data/cocostuffdataset",
}

###########
#         #
# Parsing #
#         #
###########

parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_dir",
    type=str,
    default="./datasets/coco164k",
    help="directory where to store coco files",
)
parser.add_argument(
    "--coco_stuff_dir", type=str, default="./datasets/coco164k/", help="coco_stuff directory of SPADE"
)
opt = parser.parse_args()

# convert to absolute paths
coco_stuff_dir = pathlib.Path(opt.coco_stuff_dir).resolve()
coco_stuff_dl_dir = pathlib.Path(opt.download_dir).resolve()

print("Downloading files to {}".format(coco_stuff_dl_dir))
print("Extracting to coco_stuff directory {}".format(coco_stuff_dir))

###############
#             #
# Downloading #
#             #
###############
os.makedirs(coco_stuff_dl_dir, exist_ok=True)
coco_stuff_dl_dir.mkdir(exist_ok=True)


def reporthook(chunk_number, max_chunk_size, total_size):
    percentage_dl = chunk_number * max_chunk_size / total_size * 100
    sys.stdout.write("\r{:>3d}%".format(int(percentage_dl)))
    sys.stdout.flush()


print("Downloading coco files...")
for name, url in cocofiles.items():
    print("downloading {} ...".format(name))
    dl_path = coco_stuff_dl_dir / name
    if dl_path.exists():
        print("already downloaded")
    else:
        try:
            urllib.request.urlretrieve(url + name, dl_path, reporthook)
            print("\ndone")
        except (KeyboardInterrupt, Exception) as exp:
            dl_path.unlink()
            raise exp

##############
#            #
# Extracting #
#            #
##############
print("unzipping val2017.zip")
with zipfile.ZipFile(coco_stuff_dl_dir / "val2017.zip", "r") as val_zip:
    for zip_info in val_zip.infolist():
        # remove directory, see https://stackoverflow.com/a/47632134
        if zip_info.filename[-1] == "/":
            continue
        zip_info.filename = pathlib.Path(zip_info.filename).name
        val_zip.extract(zip_info, coco_stuff_dir / "images/val2017_original")