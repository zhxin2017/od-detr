import os
import pickle
import numpy as np
from config import xml_root_dir, img_root_dir, \
    train_filelist_files, categories, img_size, train_cached_file
from functools import partial
import multiprocessing
from dataset import anno, image


h, w = img_size
category_to_idx = {name: idx + 1 for idx, name in enumerate(categories)}

# 读取文件列表
filelist = []
for filelist_file in train_filelist_files:
    with open(filelist_file, 'r') as f:
        filelist.extend([line.strip() for line in f.readlines()])

def load_data(filename):
    img_path = os.path.join(img_root_dir, f'{filename}.jpg')
    anno_path = os.path.join(xml_root_dir, f'{filename}.xml')
    boxes, cids = anno.parse_xml(anno_path, category_to_idx)
    img = image.load_image(img_path)
    img, boxes = image.resize_img(img, boxes, h, w)
    return img, boxes, cids


def cache_data():
    img_cached_data = {}
    anno_cached_data = {}
    with multiprocessing.Pool(24) as pool:
        results = pool.map(load_data, filelist)
        for filename, (img, boxes, cids) in zip(filelist, results):
            img_cached_data[filename] = img
            anno_cached_data[filename] = (boxes, cids)

    cache_data = {
        'images': img_cached_data,
        'anno': anno_cached_data
    }
    with open(train_cached_file, 'wb') as f:
        pickle.dump(cache_data, f)


if __name__ == '__main__':
    cache_data()

