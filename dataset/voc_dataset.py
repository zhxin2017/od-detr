from dataset import anno, image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
import torch


class VocDataset(Dataset):
    def __init__(self, 
                 img_root_dir, 
                 ann_root_dir, 
                 filelist_files,
                 categories,
                 input_size,
                 random_pad=False):
        self.category_to_idx = {'background': 0}
        self.category_to_idx.update({name: idx + 1 for idx, name in enumerate(categories)})   

        self.h, self.w = input_size
        self.random_pad = random_pad
        
        self.filelist = []
        for filelist_file in filelist_files:
            with open(filelist_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.filelist.append(line.strip())
        
        self.img_root_dir = img_root_dir
        self.anno_root_dir = ann_root_dir

    def __len__(self):
        return len(self.filelist)
    
    def __getitem__(self, index):
        file_name = self.filelist[index]
        img_path = f'{self.img_root_dir}/{file_name}.jpg'
        anno_path = f'{self.anno_root_dir}/{file_name}.xml'
        boxes, cids = anno.parse_xml(anno_path, self.category_to_idx)

        img = image.load_image(img_path)
        img, boxes = image.pad_img_and_boxes(img, boxes, self.h, self.w, random_pad=self.random_pad)

        return img, boxes, cids


def collate_fn(batch):
    images = [torch.tensor(item[0]) for item in batch]
    targets = [{'boxes': torch.tensor(item[1]), 'cids': torch.tensor(item[2])} for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets


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