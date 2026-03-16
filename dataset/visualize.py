import cv2
from config import categories


def draw_bbox(img, bboxes, cids, save_path=None):
    img = (img * 255).astype('uint8')
    for bbox, cid in zip(bboxes, cids):
        category = categories[cid - 1]  
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        img = cv2.putText(img, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if save_path is not None:
        cv2.imwrite(save_path, img)
    return img