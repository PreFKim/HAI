import random
import torch
import argparse
import tqdm
import os
import yaml
import numpy as np 
import math
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split

from dataset.dataset import HAI
from model.baseline import BaseWraper

def Accuracy(pred, target):
    _, pred = torch.max(pred, 1)
    return (pred == target).float().mean()

class AllLosses(nn.Module):
    def __init__(self, loss_lst=[], weights=None, names=None):
        super(AllLosses, self).__init__()
        self.loss_lst = loss_lst    
        if weights is not None:
            self.weights = weights
            if (len(loss_lst) != len(weights)):
                raise ValueError("Lenght of weights is differ from length of loss_lst")
        else : 
            self.weights = [1. for _ in range(len(loss_lst))]

        if names is not None:
            self.names = names
            if (len(loss_lst) != len(names)):
                raise ValueError("Lenght of names is differ from length of loss_lst")
        else : 
            self.names = [f"Loss {i}" for i in range(len(loss_lst))]

    def forward(self, pairs):
        ret = {
            'Total':0
            }
        # for i in range(len(self.loss_lst)):
        for i in range(len(pairs)):
            ret[self.names[i]] = self.loss_lst[i](*pairs[i]) * self.weights[i]
            ret['Total'] = ret['Total'] + ret[self.names[i]]
        return ret

class AllMetrics(nn.Module):
    def __init__(self, metric_lst=[], weights=None, names=None):
        super(AllMetrics, self).__init__()
        self.metric_lst = metric_lst    
        if weights is not None:
            self.weights = weights
            if (len(metric_lst) != len(weights)):
                raise ValueError("Lenght of weights is differ from length of metric_lst")
        else : 
            self.weights = [1. for _ in range(len(metric_lst))]

        if names is not None:
            self.names = names
            if (len(metric_lst) != len(names)):
                raise ValueError("Lenght of names is differ from length of metric_lst")
        else : 
            self.names = [f"Loss {i}" for i in range(len(metric_lst))]

    def forward(self, pairs):
        with torch.no_grad():
            ret = {
                'Total':0
                }
            # for i in range(len(self.metric_lst)):
            for i in range(len(pairs)):
                ret[self.names[i]] = self.metric_lst[i](*pairs[i]) * self.weights[i]
                ret['Total'] = ret['Total'] + ret[self.names[i]]
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

def steps(args, epoch, dataloader, model, optimizer, criterion, metric, scheduler=None, device='cpu', mode=0):

    model = model.to(device)

    desc = f"Epoch {epoch}|{'Train' if mode==0 else 'Valid'}"
    iterator = tqdm.tqdm(iterable=dataloader, desc=desc)

    summation_loss = {}
    summation_metric = {}
    
    for i, data in enumerate(iterator):
        image = data['image'].to(device) 
        target = data['target'].to(device) 
    
        pred = model(image)

        loss_pairs = [[pred, target]]
        metric_pairs = [[pred, target]]
        # if i%100==0:
        #     print(model.pose_head[0].weight)

        losses = criterion(loss_pairs)
        metrics = metric(metric_pairs)
        
        for k, v in losses.items():
            if summation_loss.get(k) is not None:
                summation_loss[k] += v.detach().cpu().numpy()
            else:
                summation_loss[k] = v.detach().cpu().numpy()

        for k, v in metrics.items():
            if summation_metric.get(k) is not None:
                summation_metric[k] += v.detach().cpu().numpy()
            else:
                summation_metric[k] = v.detach().cpu().numpy()
        
        if mode == 0:
            losses['Total'].backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        loss_info = ''
        for k, v in summation_loss.items():
            loss_info = loss_info + f'{k}:{v/(i+1)}, '
        metric_info = ''
        for k, v in summation_metric.items():
            metric_info = metric_info + f'{k}:{v/(i+1)}, '
        iterator.set_description(f"{desc}|{loss_info}|{metric_info}")
        
    return summation_loss['Total']/(i+1)
    # return {k: v/(i+1) for k, v in summation_loss.items()}

def train(args):

    print(args)

    if args.exp_name == "":
        exp_num = 0
        while True:
            save_dir = os.path.join(args.save_dir, f"{exp_num}_{args.fold_idx}")
            if os.path.exists(save_dir):
                exp_num += 1
            else:
                break
        exp_num = f"{exp_num}_{args.fold_idx}"
    else:
        exp_num = f"{args.exp_name}_{args.fold_idx}"
        save_dir = os.path.join(args.save_dir, str(exp_num))
    print("Experiments num:", exp_num)
    os.makedirs(os.path.join(save_dir, 'code'), exist_ok=True)
    arg_dict = vars(args)  
    yaml_str = yaml.dump(arg_dict, default_flow_style=False)
    with open(os.path.join(save_dir, "parameters.yaml"), 'w') as file:
        file.write(yaml_str)

    # 코드 기록
    shutil.copy('./dataset/augmentations.py', os.path.join(save_dir, 'code', 'augmentations.py'))
    shutil.copy('./dataset/dataset.py', os.path.join(save_dir, 'code', 'dataset.py'))
    shutil.copy('./model/baseline.py', os.path.join(save_dir, 'code', 'model.py'))
    shutil.copy('./train.py', os.path.join(save_dir, 'code', 'train.py'))
    shutil.copy('./cfg.py', os.path.join(save_dir, 'code', 'cfg.py'))
    
    wh = (args.image_width, args.image_height)

    train_kwargs = {
        'root_dir':'./dataset/data/train',
        'wh': wh,
        'fold_idx':args.fold_idx,
        'mode': 0,
        'aug':True,
    }
    val_kwargs= {
        'root_dir':'./dataset/data/train',
        'wh': wh,
        'fold_idx':args.fold_idx,
        'mode': 1,
        'aug':False,
    }
    
    train_set = HAI(**train_kwargs)

    valid_set = HAI(**val_kwargs)

    class_names = train_set.classes

    if args.fold_idx < 0:
        targets = [label for _, label in train_set.samples]
        # Stratified Split
        train_idx, valid_idx = train_test_split(
            range(len(targets)), test_size=0.2, stratify=targets, random_state=42
        )

        train_set.resample(train_idx)
        valid_set.resample(valid_idx)

    print(f"Num Train: {len(train_set)}, Num Valid: {len(valid_set)}")

    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    model_config = {
        'backbone_type': args.backbone_type,
        'wh' : wh,
        'num_classes':len(train_set.classes)
    }
    model = BaseWraper(**model_config).to(device)
    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warm_cosine(step, args.warmup_ratio, len(train_set)/args.batch_size * args.epochs))

    if args.checkpoint != "" :
        ckpt = torch.load(args.checkpoint, map_location=device)

        print(model.load_state_dict(ckpt['model'], strict=False))
    
    
    loss_lst = [nn.CrossEntropyLoss()]
    loss_names = [f'CE']
    loss_weights = [1.]
    
    criterion = AllLosses(loss_lst, weights=loss_weights, names=loss_names)

    metric_lst = [Accuracy]
    metric_names = ['ACC']
    metric_weights = [1.]
    metric = AllMetrics(metric_lst, weights=metric_weights, names=metric_names)

    train_losses = []
    valid_losses = []
    for i in range(args.epochs):
        model = model.train()
        train_losses.append(steps(args, i+1, train_loader, model, optimizer, criterion, metric, scheduler, device, mode=0))
        model = model.eval()
        with torch.no_grad():
            valid_losses.append(steps(args, i+1, valid_loader, model, optimizer, criterion, metric, scheduler, device, mode=1))


        ckpt = {
            "epoch": i+1,
            "config": model_config,
            "class_names": class_names,
            "model": model.state_dict(),
        }

        print(f"Epoch {i+1} Endded")
        print(f"Train_Loss: {train_losses[i]}")
        print(f"Valid_Loss: {valid_losses[i]}")

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))

        if (i+1)%5 == 0:
            torch.save(ckpt, os.path.join(save_dir, f"last_{i+1}.pt"))

        if valid_losses[-1] <= min(valid_losses):
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"Best Validation Loss {i+1} : {valid_losses[i]}")
            print(f"Saved at {os.path.join(save_dir, 'best.pt')}")


        print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="./experiments", help="")

    # 모델 파라미터
    parser.add_argument("--checkpoint", type=str, default="", help="")
    parser.add_argument("--exp_name", type=str, default="", help="")
    parser.add_argument("--fold_idx", type=int, default="", help="")
    parser.add_argument("--image_width", type=int, default=256, help="")
    parser.add_argument("--image_height", type=int, default=192, help="")
    parser.add_argument("--backbone_type", type=str, default='r152', help="")

    # 하이퍼 파라미터
    parser.add_argument("--num_workers", type=int, default=16, help="")
    parser.add_argument("--random_seed", type=int, default=42, help="Fix random seed")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--epochs", type=int, default=100, help="") 
    parser.add_argument("--batch_size", type=int, default=32, help="") 
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--gpu", type=int, default=0, help="")

    args = parser.parse_args()

    set_seed(args.random_seed)
    train(args)
    
    
    
