import cv2
from xml.etree import ElementTree as ET
import numpy as np
import random


def load_image(image_path):
    return cv2.imread(image_path) / 255

def pad_img_and_boxes(img, boxes, dst_h, dst_w):
    '''
    Args:
        img: [H, W, C], np.ndarray
        boxes: [N, 4], np.ndarray, (x1, y1, x2, y2)
    '''
    h, w, _ = img.shape
    if h / dst_h > w / dst_w:
        scale = dst_h / h
        h_ = dst_h
        w_ = int(w * scale)
    else:
        scale = dst_w / w
        h_ = int(h * scale)
        w_ = dst_w
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
    x_offset = random.randint(0, dst_w - w_)
    y_offset = random.randint(0, dst_h - h_)
    img = cv2.resize(img, (w_, h_))
    if len(boxes) > 0:
        boxes = boxes * scale + np.array([[x_offset, y_offset, x_offset, y_offset]])
    canvas[y_offset:y_offset + h_, x_offset:x_offset + w_] = img
    return canvas, boxes