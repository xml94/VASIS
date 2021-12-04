#!/usr/bin/env bash

dir=$1

find "$dir/leftImg8bit" -maxdepth 3 -name "$dir/*_leftImg8bit.png" | sort > val_images.txt
find "$dir/gtFine/val" -maxdepth 3 -name "$dir/*_trainIds.png" | sort > val_labels.txt
