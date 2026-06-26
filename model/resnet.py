import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, layers=50):
        super(ResNet, self).__init__()
        if layers==50:
            model = models.resnet50(pretrained=True)
        elif layers==152:
            model = models.resnet152(pretrained=True)
        
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.feature_dim = model.fc.in_features 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
