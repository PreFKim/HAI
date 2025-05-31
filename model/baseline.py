import torch
import torch.nn as nn
import torchvision.models as models

class BaseWraper(nn.Module):
    def __init__(self, backbone_type, wh, num_classes):
        super().__init__()

        self.wh = wh
        if backbone_type=='r50':
            self.backbone = models.resnet50(pretrained=True)
        elif backbone_type=='r152':
            self.backbone = models.resnet152(pretrained=True)
        self.feature_dim = self.backbone.fc.in_features 
        self.backbone.fc = nn.Identity()  
        self.head = nn.Linear(self.feature_dim, num_classes)  

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x