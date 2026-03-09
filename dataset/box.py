import torch


def box_cxcywh_to_xyxy(boxes):
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w = boxes[..., 2]
    h = boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def iouloss(boxes1, boxes2):
    min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])
    max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_wh = (max_xy - min_xy).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-5)
    return 1 - iou



if __name__ == '__main__':
    cxcy = torch.rand((100, 2))
    wh = torch.rand((100, 2))
    boxes = torch.cat([cxcy, wh], dim=-1)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    boxes1 = boxes_xyxy[:50]
    boxes2 = boxes_xyxy[50:]
    loss = iouloss(boxes1, boxes2)
    print(boxes1.shape, boxes2.shape, loss.shape)