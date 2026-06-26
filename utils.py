
import torch
import torch.nn as nn
import math
import random
import os
import numpy as np
import cv2
from model.baseline import BaseWraper
from contextlib import nullcontext

def Accuracy(pred, target):
    _, pred = torch.max(pred, 1)
    return (pred == target).float().mean()

class MixFunc(nn.Module):
    def __init__(self, func=nn.CrossEntropyLoss()):
        super().__init__()
        self.func = func
    
    def forward(self, x, y1, y2=None, w=1):
        if y2 is None:
            return self.func(x, y1)
        else:
            return self.func(x, y1)*w + self.func(x, y2)*(1-w)

def curriculum_scheduler(epoch, max_epoch):
    return min(1, epoch/((max_epoch)/2))

class AllEvaluators(nn.Module):
    def __init__(self, func_lst=[], weights=None, names=None, is_metric=False):
        super(AllEvaluators, self).__init__()
        self.func_lst = func_lst
        self.is_metric = is_metric  # Loss인지 Metric인지 구분

        if weights is not None:
            if len(func_lst) != len(weights):
                raise ValueError("Length of weights is different from length of func_lst")
            self.weights = weights
        else:
            self.weights = [1.0 for _ in range(len(func_lst))]

        if names is not None:
            if len(func_lst) != len(names):
                raise ValueError("Length of names is different from length of func_lst")
            self.names = names
        else:
            prefix = "Metric" if is_metric else "Loss"
            self.names = [f"{prefix} {i}" for i in range(len(func_lst))]

    def forward(self, pairs):
        context = torch.no_grad() if self.is_metric else nullcontext()

        with context:
            ret = {'Total': 0}
            for i in range(len(pairs)):
                val = self.func_lst[i](*pairs[i]) * self.weights[i]
                ret[self.names[i]] = val
                ret['Total'] += val
        return ret


def set_seed(seed):
    if seed >= 0:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def warm_cosine(step: int, warmup_ratio: int, total_steps: int) -> float:
    # Linear warmup
    warmup_steps = int(total_steps*warmup_ratio)
    if step < warmup_steps:
        return float(step) / float(max(1.0, warmup_steps))
    
    # Cosine decay
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

def load_model(model_path, device='cpu', ema=False):
    print("Model path:", model_path)
    w = torch.load(model_path, map_location=device)
    if ema and w.get('ema_model') is not None:
        weight = w['ema_model']
        print("Apply EMA weight")
    else:
        weight = w['model']
        print("Apply normal weight")
    model_config = w['config']
    class_names = w['class_names']

    converted = {}
    for k, v in weight.items():
        if k == 'head.weight':
            converted['head.0.weight'] = v
        elif k == 'head.bias':
            converted['head.0.bias'] = v
        else : 
            converted[k] = v

    print(f"Trained Epoch: {w['epoch']}")
    print("Model Config:")
    print(model_config)
    # print("Class names")
    # print(len(class_names), class_names)

    model = BaseWraper(**model_config).to(device)

    print(model.load_state_dict(converted, strict=False))
    return model, class_names

def imread_unicode(path):
    with open(path, 'rb') as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img