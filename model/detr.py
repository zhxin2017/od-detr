import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.transformer import EncLayer, DecLayer
from config import img_size, categories


def sinusoidal_pe(y, x, d_model):
    b, lq = y.shape[0], y.shape[1]
    pe = torch.zeros(b, lq, d_model, device=y.device)
    max_freq = 256
    freq = torch.exp(
        torch.arange(0, d_model//2, device=y.device, requires_grad=False) * math.log(max_freq) / (d_model//2)
    )
    pe[:, :, 0::2] = torch.sin(x.unsqueeze(2) * 2 * math.pi * freq)
    pe[:, :, 1::2] = torch.cos(y.unsqueeze(2) * 2 * math.pi * freq)
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

        # Convolutional layers to create patches and downsample 1/16
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(256, d_model, kernel_size=2, stride=2, padding=0)
        
        # Transformer encoder layers
        self.enc_layers = nn.ModuleList([EncLayer(d_model, nhead) for _ in range(num_layers)])
        
    
    def forward(self, x):
        """
        Args:
            x: input image (B, H, W, 3)
        Returns:
            encoder output (B, H*W/256, d_model) with 1/16 downsample
        """
        x = x.permute(0, 3, 1, 2)  # (B, 3, H, W)
        # Extract features through conv layers (1/16 downsample)
        x = torch.nn.functional.gelu(self.conv1(x))  # (B, 64, H/2, W/2)
        x = torch.nn.functional.gelu(self.conv2(x))  # (B, 128, H/4, W/4)
        x = torch.nn.functional.gelu(self.conv3(x))  # (B, 256, H/8, W/8)
        x = torch.nn.functional.gelu(self.conv4(x))  # (B, 256, H/16, W/16)
        
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.view(B, C, -1).transpose(1, 2)  # (B, H*W, d_model)
        
        # Apply transformer encoder layers with positional embeddings
        for enc_layer in self.enc_layers:
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
        
        self.ref_points = nn.Parameter(torch.rand(num_queries, 2))
        self.wh_embed = nn.Linear(2, d_model)
        self.dec_layers = nn.ModuleList([DecLayer(d_model, nhead) for _ in range(num_layers)])
        
        self.class_reg = nn.Linear(d_model, num_classes)
        self.xy_delta_reg = nn.ModuleList()
        self.wh_reg = nn.Linear(d_model, 2)
        self.wh_delta_reg = nn.ModuleList()
        for i in range(num_layers):
            self.xy_delta_reg.append(nn.Linear(d_model, 2))
            if i >= 1:
                self.wh_delta_reg.append(nn.Linear(d_model, 2))
    
    def forward(self, mem):

        B = mem.shape[0]
        
        mem_with_pos = mem + self.pos_embed
        
        # Store predictions
        class_preds = []
        bbox_preds = []
        pred_ref_points = self.ref_points.clone() # (num_queries, 2)
        pred_ref_points = pred_ref_points.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, 2)
        ref_point_embed = sinusoidal_pe(pred_ref_points[..., 1], pred_ref_points[..., 0], self.d_model)  # (num_queries, d_model)
        queries = ref_point_embed  # (B, num_queries, d_model)
        # Decoder layers with iterative refinement
        for layer_idx in range(len(self.dec_layers)):
            # Apply decoder layer (self-attn + cross-attn + FFN)
            queries = self.dec_layers[layer_idx](queries, mem_with_pos, mem, skip_sa=layer_idx==0)
            if layer_idx == 0:
                wh = torch.sigmoid(self.wh_reg(queries))  # (B, num_queries, 2)
            else:
                wh_delta = torch.tanh(self.wh_delta_reg[layer_idx-1](queries)) / 2.0 # (B, num_queries, 2)
                wh = wh + wh_delta
            xy_delta = torch.tanh(self.xy_delta_reg[layer_idx](queries)) / 2.0  # (B, num_queries, 2)
            pred_ref_points = pred_ref_points + xy_delta  # (num_queries, 2)
            xy_emb = sinusoidal_pe(pred_ref_points[..., 1], pred_ref_points[..., 0], self.d_model)  # (num_queries, d_model)
            wh_emb = self.wh_embed(wh)
            queries = queries + xy_emb + wh_emb
            bbox_preds.append(torch.cat([pred_ref_points, wh], dim=-1))  # (B, num_queries, 4)

            class_pred = self.class_reg(queries)  # (B, num_queries, num_classes)
            class_preds.append(class_pred)

        return class_preds, bbox_preds



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
        class_logits, bbox_preds = self.decoder(enc_output)
        
        return class_logits, bbox_preds


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
