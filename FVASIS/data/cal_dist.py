import numpy as np
import cv2
import torch


def cal_dist_from_img_name(names, dataset_mode='cityscapes'):
    if dataset_mode == 'cityscapes':
        pass
    elif dataset_mode == 'ade20k':
        pass
    elif dataset_mode == 'cocostuff':
        pass
    else:
        raise NotImplementedError('Please check the dataset mode to compute the relative distance')

def cal_connectedComponents(mask, normal_mode='norm'):
    label_idxs = np.unique(mask)
    H, W = mask.shape
    out_h_offset = np.float32(np.zeros_like(mask))
    out_w_offset = np.float32(np.zeros_like(mask))
    for label_idx in label_idxs:
        if label_idx == 0:
            continue
        tmp_mask = np.float32(mask.copy())
        tmp_mask[tmp_mask!=label_idx] = -1
        tmp_mask[tmp_mask==label_idx] = 255
        tmp_mask[tmp_mask==-1] = 0
        _, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(tmp_mask))
        connected_numbers = len(centroids)-1
        for c_idx in range(1,connected_numbers+1):
            tmp_labels = np.float32(labels.copy())
            tmp_labels[tmp_labels!=c_idx] = 0
            tmp_labels[tmp_labels==c_idx] = 1
            h_offset = (np.repeat(np.array(range(H))[...,np.newaxis],W,1) - centroids[c_idx][1])*tmp_labels
            w_offset = (np.repeat(np.array(range(W))[np.newaxis,...],H,0) - centroids[c_idx][0])*tmp_labels
            h_offset = normalize_dist(h_offset, normal_mode)
            w_offset = normalize_dist(w_offset, normal_mode)
            out_h_offset += h_offset
            out_w_offset += w_offset

    return out_h_offset, out_w_offset

def normalize_dist(offset, normal_mode):
    if normal_mode == 'no':
        return offset
    else:
        return offset / np.max(np.abs(offset)+1e-5)