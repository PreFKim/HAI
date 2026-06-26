import torch.nn as nn
import torch
from FlashInternImage.classification.models.build import build_model
from FlashInternImage.classification.config import get_config
from model.cfg import CFG


scale_config = {
    't':'flash_intern_image_t_1k_224',
    's':'flash_intern_image_s_1k_224',
    'b':'flash_intern_image_b_1k_224',
    # 'l':'flash_intern_image_l_22kto1k_384', # Custom Cuda 문제로 학습 불가능
}

class FlashInternimage(nn.Module):
    def __init__(self, scale='b'):
        super().__init__()
        self.scale = scale
        self.config = CFG(f"./FlashInternImage/classification/configs/{scale_config[scale]}.yaml")
        self.model = build_model(get_config(self.config))
        self.model.load_state_dict(torch.load(f"./weights/{scale_config[scale]}.pth")['model'], strict=False)
    
    def forward(self, x):
        x = self.model.patch_embed(x)
        N, H, W, C = x.shape
        x = x.view(N, H*W, C)

        shape=(H, W)
        seq_out = []
        for level_idx, level in enumerate(self.model.levels):
            old_shape = shape
            x, shape = level(x, shape=shape)   
        h, w = shape
        x = x.view(N, h, w, -1)
        x = self.model.conv_head(x.permute(0, 3, 1, 2))
        return x