import os
import numpy as np
import pickle
from config import num_query
from dataset import anno, image, data_cache
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch
import time


class VocDataset(Dataset):
    def __init__(self, 
                 img_root_dir, 
                 ann_root_dir, 
                 filelist_files,
                 categories,
                 input_size,
                 random_pad=False,
                 cached_file=None,
                 cached_anno_file=None
                 ):
        self.category_to_idx = {'background': 0}
        self.category_to_idx.update({name: idx + 1 for idx, name in enumerate(categories)})   

        self.h, self.w = input_size
        self.random_pad = random_pad
        self.cached_file = cached_file

        self.filelist = []
        for filelist_file in filelist_files:
            with open(filelist_file, 'r') as f:
                lines = f.readlines()
                files = [line.strip() for line in lines]
                self.filelist.extend(files)
        
        self.img_root_dir = img_root_dir
        self.ann_root_dir = ann_root_dir

        if cached_file is not None:
            assert os.path.exists(cached_file), f"Cached file {cached_file} does not exist."
            print(f"Loading cached data from {cached_file}...")
            load_start = time.time()
            with open(cached_file, 'rb') as f:
                cache_data = pickle.load(f)
            load_end = time.time()
            print(f"Loaded cached data in {load_end - load_start:.3f} seconds.")
            self.img_cache = cache_data['images']
            self.anno_cache = cache_data['anno']
        else:
            self.img_cache = None


    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        file_name = self.filelist[index]
        fetch_start = time.time()
        if self.img_cache is not None:
            img = self.img_cache[file_name]
            boxes, cids = self.anno_cache[file_name]
            boxes = boxes.astype(np.float32)
            # print(f'dtype of cached image: {img.dtype}, dtype of cached boxes: {boxes.dtype}, dtype of cached cids: {cids.dtype}')
        else:
            img_path = os.path.join(self.img_root_dir, f'{file_name}.jpg')
            anno_load_start = time.time()
            anno_path = os.path.join(self.ann_root_dir, f'{file_name}.xml')
            boxes, cids = anno.parse_xml(anno_path, self.category_to_idx)
            img = image.load_image(img_path)
            img, boxes = image.resize_img(img, boxes, self.h, self.w)
        n = len(boxes)
        boxes_padding = np.zeros((num_query - n, 4), dtype=np.float32)
        cids_padding = np.zeros((num_query - n,), dtype=np.int64)
        if n == 0:
            boxes = boxes_padding
            cids = cids_padding
        else:
            boxes = np.concatenate([boxes, boxes_padding], axis=0)
            cids = np.concatenate([cids, cids_padding], axis=0)
        img, boxes = image.pad_img_and_boxes(img, boxes, self.h, self.w, random_pad=self.random_pad)

        return img, boxes, cids


if __name__ == '__main__':
    from dataset import visualize
    img_root_dir = '../dataset/VOC2012/JPEGImages'
    xml_root_dir = '../dataset/VOC2012/Annotations'
    filelist_file = '../dataset/VOC2012/ImageSets/Main/val2017.txt'
    with open('dataset/categories.txt', 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    voc_dataset = VocDataset(img_root_dir, xml_root_dir, [filelist_file], categories, (256, 256))
    for i in range(len(voc_dataset)):
        img, boxes, cids = voc_dataset[i]
        visualize.draw_bbox(img, boxes, 'vis.jpg')
        # break