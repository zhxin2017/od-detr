import cv2
from xml.etree import ElementTree as ET
import numpy as np
import random


def load_image(image_path):
    return cv2.imread(image_path)

def resize_img(img, boxes, dst_h, dst_w):
    '''
    Args:
        img: [H, W, C], np.ndarray
        boxes: [N, 4], np.ndarray, (x1, y1, x2, y2)
        dst_h: int, destination height
        dst_w: int, destination width
    Returns:
        resized_img: [H', W', C], np.ndarray
        adjusted_boxes: [N, 4], np.ndarray
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
    resized_img = cv2.resize(img, (w_, h_))
    if len(boxes) > 0:
        adjusted_boxes = boxes * scale
    else:
        adjusted_boxes = boxes
    return resized_img, adjusted_boxes

def pad_img_and_boxes(img, boxes, dst_h, dst_w, random_pad=False):
    '''
    Args:
        img: [H, W, C], np.ndarray
        boxes: [N, 4], np.ndarray, (x1, y1, x2, y2)
    '''
    img, boxes = resize_img(img, boxes, dst_h, dst_w)
    h, w, _ = img.shape
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    if random_pad:
        x_offset = random.randint(0, dst_w - w)
        y_offset = random.randint(0, dst_h - h)
    else:
        x_offset = (dst_w - w) // 2
        y_offset = (dst_h - h) // 2
    if len(boxes) > 0:
        boxes = boxes + np.array([[x_offset, y_offset, x_offset, y_offset]], dtype=np.float32)
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return canvas, boxes