import os

import torchvision
from dataset.box import box_cxcywh_to_xyxy
import torch
from torch import optim
from model import detr
from config import batch_size, num_enc_layer, num_dec_layer, dmodel, nhead, \
    num_query, categories, img_root_dir, xml_root_dir, train_filelist_files, img_size, \
        epoch, device_name, resume, train_cached_file
from dataset.voc_dataset import VocDataset
import time
import numpy as np
from torch.utils.data import DataLoader

from train import match, eval

num_classes = len(categories) + 1  # +1 for background
img_h, img_w = img_size


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False

def main():

    detr_model = detr.DETR(dmodel, nhead, num_enc_layer, num_dec_layer, num_query, num_classes)

    optimizer = optim.Adam(detr_model.parameters(), lr=1e-4)
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device(device_name)
    detr_model.to(device)
    dataset = VocDataset(img_root_dir, xml_root_dir, train_filelist_files, categories, 
                        [img_h, img_w], random_pad=True, cached_file=train_cached_file)
    # dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=0, \
    #                         pin_memory=False)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=8, prefetch_factor=2,\
                            pin_memory=True, persistent_workers=True)
    num_batches = len(dataloader)

    if resume is not None:
        detr_model.load_state_dict(torch.load(resume, map_location=device))
        starting_epoch = int(resume.split('_epoch')[1].split('_batch')[0]) - 1
        iteration = int(resume.split('_batch')[1].split('.pth')[0])
        if iteration == num_batches:
            starting_epoch = starting_epoch + 1
        print(f"Resuming training from epoch {starting_epoch} using checkpoint: {resume}")
    else:
        starting_epoch = 0

    cls_loss_fn = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    box_resize_factor = torch.tensor([img_w, img_h, img_w, img_h], device=device)
    box_resize_factor = box_resize_factor.view([1, 1, 4])
    def train_one_epoch(e):
        iter_end = time.time()
        for j, (imgs, boxes_gt, cids_gt) in enumerate(dataloader):
            iter_start = time.time()
            data_time = iter_start - iter_end
            imgs = imgs.to(device) / 255.0
            boxes_gt = boxes_gt.to(device)
            cids_gt = cids_gt.to(device)
            n = len(boxes_gt)
            boxes_gt = boxes_gt / box_resize_factor

            cls_logits, boxes_pred_cxcywh = detr_model(imgs)
            boxes_pred = box_cxcywh_to_xyxy(boxes_pred_cxcywh)
            logits_std = cls_logits.std(dim=0).mean()
            box_std = boxes_pred.std(dim=0).mean()
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
            cids_gt = cids_gt[(gt_matched_indices_batch, gt_matched_indices_query)]

            cls_logits = cls_logits.view(n * num_query, -1)
            cls_loss = cls_loss_fn(cls_logits, cids_gt)
            cls_loss = cls_loss  * query_pos_mask + cls_loss * (1 - query_pos_mask) * 0.1
            cls_loss = cls_loss.mean()

            cls_pred = cls_logits.argmax(dim=-1)
            accu, recall, f1, n_tp = eval.eval_pred(cls_pred, cids_gt, query_pos_mask)

            # box loss
            boxes_pred = boxes_pred.view(n * num_query, -1)
            boxes_gt = boxes_gt[(gt_matched_indices_batch, gt_matched_indices_query)].view(n * num_query, -1)
            iou_loss = torchvision.ops.distance_box_iou_loss(boxes_pred, boxes_gt)
            l1_loss = torch.nn.functional.l1_loss(boxes_pred, boxes_gt, reduction='none').sum(dim=-1)
            box_loss = iou_loss + l1_loss * 5
            box_loss = box_loss * query_pos_mask
            box_loss = box_loss.sum() / (n_pos + 1e-5)

            loss = cls_loss + box_loss * 0.1
            optimizer.zero_grad()
            t = time.time()
            loss.backward()
            t_bp = time.time() - t
            # nn.utils.clip_grad_value_(tsfm.parameters(), 0.05)
            optimizer.step()
            iter_end = time.time()
            iter_time = iter_end - iter_start

            if (j + 1) % 1000 == 0 or j == num_batches - 1:
                torch.save(detr_model.state_dict(), os.path.join('checkpoints', f'detr_epoch{e + 1}_batch{j + 1}.pth'))

            print(f'|e {e + 1}/{epoch}|b {j + 1}/{num_batches}|'
                    f'cl {cls_loss.detach().item():.3f}|'
                    f'bl {box_loss.detach().item():.3f}|'
                    f'ac {accu:.3f}|rc {recall:.3f}: {n_tp}/{n_pos}|'
                    f'bp {t_bp:.3f}s|iter {iter_time:.3f}s|data {data_time:.3f}s|'
                    f'lgt_std {logits_std:.3f}|box_std {box_std:.3f}')
    for i in range(epoch):
        if i < starting_epoch:
            continue
        train_one_epoch(i)
    torch.save(detr_model.state_dict(), os.path.join('checkpoints', f'detr_epoch{i + 1}.pth'))

if __name__ == '__main__':
    main()
