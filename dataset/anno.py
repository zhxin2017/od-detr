
from xml.etree import ElementTree as ET
import numpy as np


def parse_xml(xml_path, category_to_idx):
    with open(xml_path, 'r') as xml:
        data = xml.read()
    root = ET.XML(data)
    objs = root.findall('object')
    boxes = []
    cids = []
    for obj in objs:
        name = obj.find('name').text
        if name not in category_to_idx:
            continue
        cid = category_to_idx[name]
        cids.append(cid)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    boxes = np.array(boxes)
    cids = np.array(cids)
    return boxes, cids

