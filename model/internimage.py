from InternImage.classification.models import build_model
from InternImage.classification.config import get_config
import torch
import torch.nn as nn
from model.cfg import CFG

scale_config = {
    't':'internimage_t_1k_224',
    's':'internimage_s_1k_224',
    'b':'internimage_b_1k_224',
    'l':'internimage_l_22kto1k_384',
    'xl':'internimage_xl_22kto1k_384',
    'h':'internimage_h_22kto1k_640',
    'g':'internimage_g_22kto1k_512',
}

class Internimage(nn.Module):
    def __init__(self, scale='b'):
        super().__init__()
        self.scale = scale
        self.config = CFG(f"./InternImage/classification/configs/{scale_config[scale]}.yaml")
        self.model = build_model(get_config(self.config))
        self.model.load_state_dict(torch.load(f"./weights/{scale_config[scale]}.pth")['model'], strict=False)
        
    
    def forward(self, x):
        if self.model.use_clip_projector: # for InternImage-H/G
            x = self.model.forward_clip_projector(x)
        else: # for InternImage-T/S/B/L/XL
            x = self.model.forward_features(x)
        return x # Pooled_feature