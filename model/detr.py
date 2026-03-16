import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from model.transformer import EncLayer, DecLayer
from config import img_size, categories


def sinusoidal_pe(y, x, d_model):
    x = x.detach()
    y = y.detach()
    b, lq = y.shape[0], y.shape[1]
    pe = torch.zeros(b, lq, d_model, device=y.device)
    max_freq = 256
    freq = torch.exp(
        torch.arange(0, d_model//4, device=y.device) * math.log(max_freq) / (d_model//4)
    )
    
    # Properly encode 2D positions with sin and cos for both x and y
    for i in range(d_model // 4):
        pe[:, :, 4*i] = torch.sin(x * freq[i])
        pe[:, :, 4*i+1] = torch.cos(x * freq[i])
        pe[:, :, 4*i+2] = torch.sin(y * freq[i])
        pe[:, :, 4*i+3] = torch.cos(y * freq[i])
    
    return pe


class Encoder(nn.Module):
    """DETR-like encoder with conv layers and transformer blocks"""
    def __init__(self, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        h_fm, w_fm = img_size[0] // 16, img_size[1] // 16
        x_coords = torch.arange(w_fm).unsqueeze(0).repeat(h_fm, 1) / w_fm
        y_coords = torch.arange(h_fm).unsqueeze(1).repeat(1, w_fm) / h_fm
        x_coords = x_coords.view(1, -1)
        y_coords = y_coords.view(1, -1)
        self.pe = sinusoidal_pe(y_coords, x_coords, d_model)
        self.register_buffer('pos_embed', self.pe)

        # use resnet50 backbone for better feature extraction
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # only use the third conv layer output (1/16 downsample) to reduce memory and speed up training
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.conv_proj = nn.Conv2d(1024, d_model, kernel_size=1)  # Project to d_model
        
        # Transformer encoder layers
        self.enc_layers = nn.ModuleList([EncLayer(d_model, nhead) for _ in range(num_layers)])
        # self.ln_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
    
    def forward(self, x):
        """
        Args:
            x: input image (B, H, W, 3)
        Returns:
            encoder output (B, H*W/256, d_model) with 1/16 downsample
        """
        # bgr from openCV to rgb
        x = x[:, :, :, [2, 1, 0]]
        x = x.permute(0, 3, 1, 2)  # (B, 3, H, W)
        # normalize with imagenet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        # Extract features through conv layers (1/16 downsample)
        x = self.backbone(x)  # (B, 1024, H/16, W/16)
        x = self.conv_proj(x)  # (B, d_model, H/16, W/16)

        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, d_model)
        
        # Apply transformer encoder layers with positional embeddings
        for i, enc_layer in enumerate(self.enc_layers):
            # x = self.ln_layers[i](x)
            x_q = x + self.pos_embed 
            x_k = x + self.pos_embed
            x = enc_layer(x_q, x_k, x)
        
        return x


class Decoder(nn.Module):
    """DETR-like decoder with learnable object queries and iterative refinement"""
    def __init__(self, d_model=256, nhead=4, num_layers=6, num_queries=100, num_classes=23):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        h_fm, w_fm = img_size[0] // 16, img_size[1] // 16
        x_coords = torch.arange(w_fm).unsqueeze(0).repeat(h_fm, 1) / w_fm
        y_coords = torch.arange(h_fm).unsqueeze(1).repeat(1, w_fm) / h_fm
        x_coords = x_coords.view(1, -1)
        y_coords = y_coords.view(1, -1)
        self.pe = sinusoidal_pe(y_coords, x_coords, d_model)
        self.register_buffer('pos_embed', self.pe)
        self.mem_ln = nn.LayerNorm(d_model)
        
        self.ref_points = nn.Parameter(torch.rand(num_queries, 2))
        self.dec_layers = nn.ModuleList()
        for i in range(num_layers):
            dec_layer = DecLayer(d_model, nhead)
            self.dec_layers.append(dec_layer)
        # self.ln_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.class_reg = nn.Linear(d_model, num_classes)
        self.xy_delta_reg = nn.Linear(d_model, 2)
        self.wh = nn.Linear(d_model, 2)
    
    def forward(self, mem):

        mem = self.mem_ln(mem)
        mem_with_pos = mem + self.pos_embed
        pred_xy = self.ref_points.unsqueeze(0).expand(mem.size(0), -1, -1)  # (B, num_queries, 2)
        
        # Store predictions
        ref_point_embed = sinusoidal_pe(pred_xy[..., 1], pred_xy[..., 0], self.d_model)  # (num_queries, d_model)
        # print('Initial ref points std across batch(expected to be 0): ', ref_point_embed.std(dim=0).mean().item())
        # print('Initial ref points std across feature dim: ', ref_point_embed.std(dim=2).mean().item())
        queries = ref_point_embed  # (B, num_queries, d_model)
        # Decoder layers with iterative refinement
        for layer_idx in range(len(self.dec_layers)):
            # Apply decoder layer (self-attn + cross-attn + FFN)
            queries = self.dec_layers[layer_idx](queries, mem_with_pos, mem, skip_sa=layer_idx==0)

        wh = torch.sigmoid(self.wh(queries)) # (B, num_queries, 2)
        xy_delta = torch.tanh(self.xy_delta_reg(queries))  # (B, num_queries, 2)
        pred_xy = pred_xy + xy_delta  # (num_queries, 2)
        pred_box = torch.cat([pred_xy, wh], dim=-1)  # (B, num_queries, 4)
        class_logits = self.class_reg(queries)  # (B, num_queries, num_classes)

        return class_logits, pred_box



class DETR(nn.Module):
    """Complete DETR model with encoder and decoder"""
    def __init__(self, dmodel=256, nhead=4, enc_layers=4, dec_layers=6, 
                 num_queries=100, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = len(categories)
        
        self.encoder = Encoder(dmodel, nhead, enc_layers)
        self.decoder = Decoder(dmodel, nhead, dec_layers, num_queries, num_classes)
        self.d_model = dmodel
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) - input images
        Returns:
            class_logits: list of (B, num_queries, num_classes+1)
            bbox_preds: list of (B, num_queries, 4) for each decoder layer
            spatial_dims: (h, w) - spatial dimensions of encoder output
        """
        # Encode
        enc_output = self.encoder(images)
        
        # Decode
        class_logit, bbox_pred = self.decoder(enc_output)
        
        return class_logit, bbox_pred


if __name__ == '__main__':
    # Test the model
    model = DETR()
    dummy_input = torch.randn(2, 3, img_size[0], img_size[1])
    class_logits, bbox_preds = model(dummy_input)
    print("Class logits shapes:")
    for i, logits in enumerate(class_logits):
        print(f"Layer {i}: {logits.shape}")
    print("BBox preds shapes:")
    for i, bbox in enumerate(bbox_preds):
        print(f"Layer {i}: {bbox.shape}")
