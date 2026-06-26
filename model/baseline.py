import torch
import torch.nn as nn
from model.resnet import ResNet
from model.internimage import Internimage
from model.flashinternimage import FlashInternimage
from model.attnpool import AttentionPool2d
from transformers import AutoModel
import timm
from cfg import IN_mean, IN_std

return_type = {
    'r':'bchw',
    'v':'bld', # vit https://huggingface.co/therealcyberlord/stanford-car-vit-patch16 : apache-2.0, Stanford car dataset
    'c':'bchw', # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
    'i':'bc', # interimage https://github.com/OpenGVLab/InternImage : mit, imagenet-1k, imagenet-22k
    's':'bhwc', # swin v2 https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k : mit, imagenet-1k, imagenet-22k
    'f':'bchw', # FlashInterimage https://github.com/OpenGVLab/DCNv4?tab=readme-ov-file : mit, imagenet-1k, imagenet-22k
}

class BaseWraper(nn.Module):
    def __init__(
            self, 
            backbone_type, 
            wh, 
            num_classes=396, 
            pool_type='avg', 
            cls_token_per_class=False,
            n_linear = 0,
            layernorm = False,
            mean=IN_mean.tolist(),
            std=IN_std.tolist()
            ):
        super().__init__()

        self.wh = wh
        self.backbone_type = backbone_type
        self.return_type = return_type[backbone_type[0]]

        self.mean = mean
        self.std = std
        if backbone_type=='r50':
            self.backbone = ResNet(layers=50)
        elif backbone_type=='r152':
            self.backbone = ResNet(layers=152)
            # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
        
        elif backbone_type == 'ct': # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
            self.backbone = AutoModel.from_pretrained("facebook/convnextv2-tiny-22k-384", trust_remote_code=True)
        elif backbone_type == 'cb': # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
            self.backbone = AutoModel.from_pretrained("facebook/convnextv2-base-22k-384", trust_remote_code=True)
        elif backbone_type == 'cl': # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
            self.backbone = AutoModel.from_pretrained("facebook/convnextv2-large-22k-384", trust_remote_code=True)
        elif backbone_type == 'ch': # convnextv2 https://huggingface.co/facebook/convnextv2-huge-22k-512 : apache-2.0, ImageNet-22K
            self.backbone = AutoModel.from_pretrained("facebook/convnextv2-huge-22k-512", trust_remote_code=True)
        
        elif backbone_type == 'st': # swin v2 https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k : mit, imagenet-1k, imagenet-22k
            self.backbone = timm.create_model(
                'timm/swinv2_tiny_window8_256.ms_in1k',
                pretrained=True,
                features_only=True,
            )
        elif backbone_type == 'ss': # swin v2 https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k : mit, imagenet-1k, imagenet-22k
            self.backbone = timm.create_model(
                'timm/swinv2_small_window8_256.ms_in1k',
                pretrained=True,
                features_only=True,
            )
        elif backbone_type == 'sb': # swin v2 https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k : mit, imagenet-1k, imagenet-22k
            self.backbone = timm.create_model(
                'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
                pretrained=True,
                features_only=True,
            )
        elif backbone_type == 'sl': # swin v2 https://huggingface.co/timm/swinv2_large_window12to24_192to384.ms_in22k_ft_in1k : mit, imagenet-1k, imagenet-22k
            self.backbone = timm.create_model(
                'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
                pretrained=True,
                features_only=True,
            )
        elif backbone_type[0] == 'i':
            self.backbone = Internimage(scale=backbone_type[1:])
        elif backbone_type[0] == 'f':
            self.backbone = FlashInternimage(scale=backbone_type[1:])
        elif backbone_type == 'vn':
            self.backbone = AutoModel.from_pretrained("therealcyberlord/stanford-car-vit-patch16")
        elif backbone_type == 'vc': # ViT Clip, https://huggingface.co/tanganke/clip-vit-large-patch14_stanford-cars
            self.backbone = AutoModel.from_pretrained("tanganke/clip-vit-large-patch14_stanford-cars")
        else:
            raise NotImplementedError(f"This backbone type({backbone_type}) is not implemnted")

        self.get_last_dim()

        self.pool_type = pool_type
        if pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
            if cls_token_per_class :
                raise NotImplementedError("cls_token_per_class option should be false for avg pool")
        elif pool_type == 'attn':
            self.pool = AttentionPool2d(
                spacial_dim=self.featuremap_size[0]*self.featuremap_size[1], 
                embed_dim=self.last_dim,
                num_heads=8,
                num_cls_token=num_classes if cls_token_per_class else 1
                )
        
        sequence = []
        if layernorm:
            sequence.append(nn.LayerNorm(self.last_dim))
        for _ in range(n_linear):
            sequence.append(nn.Linear(self.last_dim, self.last_dim))
            sequence.append(nn.GELU())
        sequence.append(nn.Linear(self.last_dim, 1 if cls_token_per_class else num_classes))
        self.head = nn.Sequential(*sequence)
        # self.head = nn.Linear(self.last_dim, 1 if cls_token_per_class else num_classes)

        self.cls_token_per_class = cls_token_per_class

    def get_last_feature(self, x):
        if self.backbone_type[0] == 's':
            x = x[-1]
        elif self.backbone_type[0] in ['c', 'v']:
            x = x['last_hidden_state']

        if self.return_type == 'bhwc':
            x = torch.permute(x, (0, 3, 1, 2))
        elif self.return_type == 'bld':
            x = x[:, 0]
        return x
    
    def get_last_dim(self):
        if self.backbone_type[0] in ['i', 'f']:
            device = 'cuda'
        else:
            device = 'cpu'
        with torch.no_grad():
            x = torch.randn(1, 3, self.wh[1], self.wh[0]).to(device)
            x = self.get_last_feature(self.backbone.to(device)(x))
        
        if self.backbone_type[0] in ['i', 'v']:
            _, c = x.shape
            h, w = 1, 1
        else:
            _, c, h, w = x.shape
        self.featuremap_size = [w, h]
        self.last_dim = c


    def forward(self, x):
        output = {
            'feature':None,
            'pooled_feature':None,
            'output':None,
        }
        x = self.get_last_feature(self.backbone(x))
        output['feature'] = x
        
        if self.backbone_type[0] in ['i', 'v']:
            b, c = x.shape
        else:
            b, c, h, w = x.shape
            x = self.pool(x) # b, c, h, w -> b, c, num_classes

        if self.cls_token_per_class:
            x = torch.permute(x, (0, 2, 1)) # b, c, num_classes -> b, num_classes, c
            output['pooled_feature'] = x
            x = self.head(x).reshape(b, -1) # b, num_classes, c -> b, num_classes, 1 -> b, num_classes
        else:
            x = x.reshape(b, c) # b, c, h, w -> b, c, 1
            output['pooled_feature'] = x
            x = self.head(x) # b, num_classes
        output['output'] = x
        return output