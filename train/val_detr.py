import os
import torch
from torch.utils.data import DataLoader
from model import detr
from config import batch_size, num_enc_layer, num_dec_layer, dmodel, nhead, \
    num_query, categories, img_root_dir, xml_root_dir, img_size, val_filelist_files
from dataset.voc_dataset import VocDataset
from dataset.box import box_cxcywh_to_xyxy
from dataset import visualize
import cv2

num_classes = len(categories) + 1  # +1 for background
img_h, img_w = img_size

# Load model
detr_model = detr.DETR(dmodel, nhead, num_enc_layer, num_dec_layer, num_query, num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
detr_model.to(device)

# Load the latest checkpoint (assuming epoch6_batch500.pth is the final one)
checkpoint_path = 'checkpoints/detr_epoch18_batch4549.pth'
detr_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
detr_model.eval()

# Val dataset
val_dataset = VocDataset(img_root_dir, xml_root_dir, val_filelist_files, categories, [img_h, img_w])
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

# Create results directory
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Box resize factor for denormalization
box_resize_factor = torch.tensor([img_w, img_h, img_w, img_h], device=device).view(1, 1, 4)

# Inference threshold
conf_threshold = 0.5

with torch.no_grad():
    for i, (imgs, boxes_gt, cids_gt) in enumerate(val_dataloader):
        imgs = imgs.to(device) / 255
        file_name = val_dataset.filelist[i]  # Assuming batch_size=1 and shuffle=False

        # Forward pass
        cls_logits_multilayer, boxes_cxcxwh_pred_multilayer = detr_model(imgs)

        # Take the last layer
        cls_logits = cls_logits_multilayer[-1]  # [1, num_query, num_classes]
        boxes_pred = boxes_cxcxwh_pred_multilayer[-1]  # [1, num_query, 4]

        # Convert to xyxy
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_pred)

        # Get probabilities
        probs = torch.softmax(cls_logits, dim=-1)  # [1, num_query, num_classes]
        pred_scores, pred_classes = probs.max(dim=-1)  # [1, num_query]

        # Filter predictions
        # keep = (pred_classes != 0) & (pred_scores > conf_threshold)  # [1, num_query]
        keep = (pred_classes != 0) # [1, num_query]
        keep = keep.squeeze(0)  # [num_query]
        if keep.sum() == 0:
            print(f'No valid predictions for {file_name}, skipping visualization.')
            continue
        pred_boxes = boxes_xyxy.squeeze(0)[keep]  # [num_keep, 4]
        
        pred_classes = pred_classes.squeeze(0)[keep]
        pred_scores = pred_scores.squeeze(0)[keep]

        # Denormalize boxes
        pred_boxes = pred_boxes * box_resize_factor.squeeze(0)

        # print(pred_boxes)

        # Get image for visualization (padded)
        img = imgs.squeeze(0).cpu().numpy()  # [H, W, 3]

        # Draw boxes
        bboxes = pred_boxes.cpu().numpy()
        vis_img = visualize.draw_bbox(img, bboxes, pred_classes.cpu().numpy())

        # Save
        save_path = os.path.join(results_dir, f'{file_name}.jpg')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f'Saving visualization to {save_path}')
        cv2.imwrite(save_path, vis_img)

        if i % 100 == 0:
            print(f'Processed {i+1}/{len(val_dataset)} images')
