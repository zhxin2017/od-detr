import os

import torchvision
from dataset.box import box_cxcywh_to_xyxy, iouloss
import torch
from torch import optim
from model import detr
from config import batch_size, num_enc_layer, num_dec_layer, dmodel, nhead, \
    num_query, categories, img_root_dir, xml_root_dir, filelist_files, img_size, \
        epoch, device
from dataset.voc_dataset import VocDataset, collate_fn
import time
import numpy as np
from torch.utils.data import DataLoader

from train import match, eval

num_classes = len(categories) + 1  # +1 for background
img_h, img_w = img_size

detr = detr.DETR(dmodel, nhead, num_enc_layer, num_dec_layer, num_query, num_classes)

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

optimizer = optim.Adam(detr.parameters(), lr=1e-5)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device(device)
detr.to(device)
dataset = VocDataset(img_root_dir, xml_root_dir, filelist_files, categories, [img_h, img_w])
dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
num_batches = len(dataloader)

cls_loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

box_resize_factor = torch.tensor([img_w, img_h, img_w, img_h], device=device)
box_resize_factor = box_resize_factor.view([1, 1, 4])
def train_one_epoch(e):
    for j, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        n = len(targets)
        cids_gt = []
        boxes_gt = []
        for k in range(n):
            # 优化：一次to()转移同时指定dtype，避免重复转移
            cids_per_img = targets[k]['cids'].to(device=device, dtype=torch.long)
            pad_num = num_query - len(cids_per_img)
            cids_padding_per_img = torch.zeros(pad_num, dtype=torch.long, device=device)
            cids_per_img = torch.cat([cids_per_img, cids_padding_per_img], dim=0)
            cids_gt.append(cids_per_img)
            boxes_per_img = targets[k]['boxes'].to(device=device, dtype=torch.float)
            boxes_padding_per_img = torch.zeros(pad_num, 4, dtype=torch.float, device=device)
            boxes_per_img = torch.cat([boxes_per_img, boxes_padding_per_img], dim=0)
            boxes_gt.append(boxes_per_img)

        cids_gt = torch.stack(cids_gt, dim=0)
        boxes_gt = torch.stack(boxes_gt, dim=0)
        boxes_gt = boxes_gt / box_resize_factor

        cls_logits_multilayer, boxes_cxcxwh_pred_multilayer = detr(imgs)
        boxes_xyxy_pred_multilayer = [box_cxcywh_to_xyxy(boxes_cxcxwh_pred) for boxes_cxcxwh_pred in boxes_cxcxwh_pred_multilayer]
        boxes_pred = boxes_xyxy_pred_multilayer[-1]
        cls_logits = cls_logits_multilayer[-1]
        gt_pos_mask = (cids_gt > 0)
        gt_pos_mask = gt_pos_mask.view(n, 1, num_query) * 1

        rows, cols = match.assign_query(boxes_gt, boxes_pred, cids_gt, cls_logits, gt_pos_mask)

        cols = torch.tensor(np.stack(cols), device=device)

        num_pos_cls = torch.sum(gt_pos_mask, dim=-1)

        gt_matched_indices_batch = torch.arange(n, device=device).view(n, 1). \
            expand(n, num_query).contiguous().view(n * num_query)

        gt_matched_indices_query = cols.view(n * num_query)

        query_pos_mask = (cols < num_pos_cls).view(n * num_query) * 1
        n_pos = query_pos_mask.sum()

        # cls loss
        cls_losses = []
        cls_total_loss = 0
        cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]
        for layer_idx in range(num_dec_layer):
            loss_weight = 0.5 ** (num_dec_layer - 1 - layer_idx)
            cls_logits = cls_logits_multilayer[layer_idx]
            cls_logits = cls_logits.view(n * num_query, -1)
            cls_loss = cls_loss_fn(cls_logits, cids_gt)
            cls_loss = cls_loss.mean() * loss_weight
            cls_total_loss = cls_total_loss + cls_loss
            cls_losses.append(cls_loss)

        cls_pred = cls_logits.argmax(dim=-1)
        accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt, query_pos_mask)

        # box loss
        box_losses = []
        box_total_loss = 0
        boxes_gt = boxes_gt[(gt_matched_indices_batch, gt_matched_indices_query)].view(n * num_query, -1)
        for layer_idx in range(num_dec_layer):
            loss_weight = 0.5 ** (num_dec_layer - 1 - layer_idx)
            boxes_pred_layer = boxes_xyxy_pred_multilayer[layer_idx].view(n * num_query, -1)
            iou_loss = iouloss(boxes_pred_layer, boxes_gt)
            l1_loss = torch.norm(boxes_pred_layer - boxes_gt, p=1, dim=-1)
            box_loss = iou_loss + l1_loss
            box_loss = box_loss * query_pos_mask
            box_loss = box_loss.sum() / (n_pos + 1e-5)
            box_total_loss = box_total_loss + box_loss * loss_weight
            box_losses.append(box_loss)

        loss = cls_total_loss * 5 + box_total_loss
        optimizer.zero_grad()
        t = time.time()
        loss.backward()
        t_bp = time.time() - t
        # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
        optimizer.step()

        if j % 1 == 0 or j == num_batches - 1:
            torch.save(detr.state_dict(), os.path.join('checkpoints', f'detr_epoch{e}_batch{j}.pth'))

        print(f'|epoch {e + 1}/{epoch}|batch {j}/{num_batches}|'
                f'cl {cls_total_loss.detach().item():.3f}|'
                f'bl {box_total_loss.detach().item():.3f}|'
                f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|'
                f'bp {t_bp:.3f}s|')


if __name__ == '__main__':
    for i in range(epoch):
        train_one_epoch(i)
