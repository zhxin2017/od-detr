import torch
from torch import nn
import scipy
import torchvision.ops


def assign_query(boxes_gt, boxes_pred, cids_gt, cls_pred, gt_pos_mask):
    B, N, C = boxes_gt.shape
    if len(boxes_pred.shape) == 2:
        boxes_pred = boxes_pred.unsqueeze(0).repeat(B, 1, 1)
    n_pos = gt_pos_mask.sum(dim=-1).view(B)

    rows = []
    cols = []
    for i in range(B):
        if n_pos[i] == 0:
            rows.append(list(range(N)))
            cols.append(list(range(N)))
            continue
        with torch.no_grad():

            iouloss = 1 - torchvision.ops.box_iou(boxes_pred[i], boxes_gt[i, :n_pos[i]])
            l1loss = torch.cdist(boxes_pred[i], boxes_gt[i, :n_pos[i]], p=1)
            box_loss = iouloss + l1loss
            probs = torch.log_softmax(cls_pred[i], dim=-1)
            cls_loss = - probs[:, cids_gt[i, :n_pos[i]]]

            # print(box_loss.mean(), cls_loss.mean())
            # print('in match', box_loss.shape, cls_loss.shape)

            total_loss = box_loss + cls_loss
        # total_loss[total_loss == torch.nan] = 1e8
        row_, col_ = scipy.optimize.linear_sum_assignment(total_loss.detach().cpu().numpy())
        col_ = col_.tolist()
        row_ = row_.tolist()
        unmatched_col = set(range(N)) - set(col_)
        row = list(range(N))
        col = []
        for j in range(N):
            if len(row_) == 0 or j != row_[0]:
                col.append(unmatched_col.pop())
            else:
                col.append(col_.pop(0))
                row_.pop(0)

        rows.append(row)
        cols.append(col)

    return rows, cols
