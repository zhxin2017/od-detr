import cv2


def draw_bbox(img, bboxes, save_path=None):
    img = (img * 255).astype('uint8')
    for bbox in bboxes:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    if save_path is not None:
        cv2.imwrite(save_path, img)
    return img