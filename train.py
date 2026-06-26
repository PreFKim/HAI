import torch
import argparse
import tqdm
import os
import yaml
import numpy as np 
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split

from torch.cuda.amp import GradScaler, autocast

from dataset.dataset import HAI
from dataset.augmentations import cutmix_data, mixup_data
from model.baseline import BaseWraper
from ema import EMA
from utils import Accuracy, MixFunc, AllEvaluators, set_seed, warm_cosine, curriculum_scheduler
from cfg import IN_mean, IN_std, H_mean, H_std


def steps(args, epoch, dataloader, model, optimizer, criterion, metric, scaler, scheduler=None, ema=None, device='cpu', mode=0, difficulty=1.):

    model = model.to(device)

    desc = f"Epoch {epoch}|{'Train' if mode==0 else 'Valid'}"
    iterator = tqdm.tqdm(iterable=dataloader, desc=desc)

    summation_loss = {}
    summation_metric = {}
    
    for i, data in enumerate(iterator):
        image = data['image'].to(device) 
        target = data['target'].to(device) 

        if np.random.uniform() < args.mix_ratio*difficulty and mode == 0:
            if np.random.uniform() < args.cutmix_ratio:
                mix_image, target1, target2, lam = cutmix_data(image, target, alpha=1)
            else:
                mix_image, target1, target2, lam = mixup_data(image, target, alpha=0.8)
            with autocast():
                pred = model(mix_image)['output']
            loss_pairs = [[pred, target1, target2, lam]]
            metric_pairs = [[pred, target1, target2, lam]] 
        else:
            with autocast():
                pred = model(image)['output']
            loss_pairs = [[pred, target]]
            metric_pairs = [[pred, target]]
            
        with autocast():

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
            scaler.scale(losses['Total']).backward()
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update_weight()
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
    
    if args.erase_other_car:
        args.det_crop = True

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
    if args.normalize_type == "IN":
        mean = IN_mean
        std = IN_std
    else:
        mean = H_mean
        std = H_std
    wh = (args.image_width, args.image_height)

    train_kwargs = {
        'root_dir':'./dataset/data/train',
        'det_path':'./dataset/detection/processed/train_det.json', 
        'wh': wh,
        'fold_idx':args.fold_idx,
        'n_fold':args.n_fold,
        'det_crop':args.det_crop,
        'erase_other_car':args.erase_other_car,
        'mean':mean,
        'std':std,
        'mode': 0,
        'aug':True,
    }
    val_kwargs= {
        'root_dir':'./dataset/data/train',
        'det_path':'./dataset/detection/processed/train_det.json', 
        'wh': wh,
        'fold_idx':args.fold_idx,
        'n_fold':args.n_fold,
        'det_crop':args.det_crop,
        'erase_other_car':args.erase_other_car,
        'mean':mean,
        'std':std,
        'mode': 1,
        'aug':False,
    }
    
    train_set = HAI(**train_kwargs)

    valid_set = HAI(**val_kwargs)

    class_names = train_set.classes

    if args.fold_idx < 0:
        targets = [label for _, _, label in train_set.samples]
        # Stratified Split
        train_idx, valid_idx = train_test_split(
            range(len(targets)), test_size=0.2, stratify=targets, random_state=42
        )

        train_set.resample(train_idx)
        valid_set.resample(valid_idx)

    print(f"Num Train: {len(train_set)}, Num Valid: {len(valid_set)}")

    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    if args.gpu >= 0: # InternImage GPU 문제 때문에 설정
        torch.cuda.set_device(args.gpu)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    model_config = {
        'backbone_type': args.backbone_type,
        'wh' : wh,
        'num_classes':len(train_set.classes),
        'pool_type':args.pool_type,
        'cls_token_per_class': args.cls_token_per_class,
        'n_linear':args.n_linear,
        'layernorm':args.layernorm,
        'mean':mean.tolist(),
        'std':std.tolist(),
    }
    model = BaseWraper(**model_config).to(device)
    scaler = GradScaler()
    ema = None
    if args.ema_alpha>0:
        print("EMA Model Ready")
        ema_model = BaseWraper(**model_config).to(device)
        ema = EMA(model, ema_model, max_alpha=args.ema_alpha)
    
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: warm_cosine(step, args.warmup_ratio, len(train_set)/args.batch_size * args.epochs))

    if args.checkpoint != "" :
        ckpt = torch.load(args.checkpoint, map_location=device)

        print(model.load_state_dict(ckpt['model'], strict=False))
    
    
    loss_lst = [MixFunc(nn.CrossEntropyLoss(label_smoothing=args.label_smoothing))]
    loss_names = [f'Cutmix(CE)']
    loss_weights = [1.]
    
    criterion = AllEvaluators(loss_lst, weights=loss_weights, names=loss_names)
    
    metric_lst = [MixFunc(Accuracy)]
    metric_names = ['Cutmix(ACC)']
    metric_weights = [1.]
    metric = AllEvaluators(metric_lst, weights=metric_weights, names=metric_names, is_metric=True)

    train_losses = []
    valid_losses = []
    ema_valid_losses = []
    for i in range(args.epochs):
        if args.curriculum:
            train_set.difficulty = curriculum_scheduler(i, args.epochs)
            print(f"Curr difficulty : {train_set.difficulty}")
        model = model.train()
        train_losses.append(steps(args, i+1, train_loader, model, optimizer, criterion, metric, scaler, scheduler, ema, device, mode=0, difficulty=train_set.difficulty))
        model = model.eval()
        with torch.no_grad():
            valid_losses.append(steps(args, i+1, valid_loader, model, optimizer, criterion, metric, scaler, scheduler, ema, device, mode=1, difficulty=train_set.difficulty))

        if ema is not None:
            ema_model = ema.ema_model.eval()
            with torch.no_grad():
                ema_valid_losses.append(steps(args, i+1, valid_loader, ema_model, optimizer, criterion, metric, scaler, scheduler, ema, device, mode=1, difficulty=train_set.difficulty))
            


        ckpt = {
            "epoch": i+1,
            "config": model_config,
            "class_names": class_names,
            "model": model.state_dict(),
            **({
                "ema_model": ema.state_dict(),
                } if ema is not None else {}),
        }

        print(f"Epoch {i+1} Endded")
        print(f"Train_Loss: {train_losses[i]}")
        print(f"Valid_Loss: {valid_losses[i]}")

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))

        if (i+1)%(args.epochs//5) == 0:
            torch.save(ckpt, os.path.join(save_dir, f"last_{i+1}.pt"))

        if valid_losses[-1] <= min(valid_losses):
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"Best Validation Loss {i+1} : {valid_losses[i]}")
            print(f"Saved at {os.path.join(save_dir, 'best.pt')}")

        if ema is not None:
            if ema_valid_losses[-1] <= min(ema_valid_losses):
                torch.save(ckpt, os.path.join(save_dir, "best_ema.pt"))
                print(f"Best EMA Validation Loss {i+1} : {ema_valid_losses[i]}")
                print(f"Saved at {os.path.join(save_dir, 'best_ema.pt')}")

        print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="./experiments", help="")
    parser.add_argument("--exp_name", type=str, default="", help="")

    # 모델 파라미터
    parser.add_argument("--checkpoint", type=str, default="", help="")
    parser.add_argument("--image_width", type=int, default=512, help="")
    parser.add_argument("--image_height", type=int, default=382, help="")
    parser.add_argument("--backbone_type", type=str, default='fb', help="")
    parser.add_argument("--layernorm", action='store_true', help="")
    parser.add_argument("--n_linear", type=int, default=0, help="")
    parser.add_argument("--pool_type", type=str, default='avg', help="")
    parser.add_argument("--cls_token_per_class", action='store_true', help="")

    # 데이터 파라미터
    parser.add_argument("--normalize_type", type=str, default='IN', help="IN: ImageNet, H: Harf")
    parser.add_argument("--mix_ratio", type=float, default=0.25, help="")
    parser.add_argument("--cutmix_ratio", type=float, default=.5, help="")
    parser.add_argument("--curriculum", action='store_true', help="")
    parser.add_argument("--det_crop", action='store_true', help="")
    parser.add_argument("--erase_other_car", action='store_true', help="")
    
    # K-Fold 파라미터
    parser.add_argument("--fold_idx", type=int, default=-1, help="")
    parser.add_argument("--n_fold", type=int, default=5, help="")

    # 하이퍼 파라미터
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="")
    parser.add_argument("--num_workers", type=int, default=16, help="")
    parser.add_argument("--random_seed", type=int, default=42, help="Fix random seed")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="")
    parser.add_argument("--epochs", type=int, default=100, help="") 
    parser.add_argument("--batch_size", type=int, default=32, help="") 
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="")
    parser.add_argument("--ema_alpha", type=float, default=0., help="")
    parser.add_argument("--gpu", type=int, default=0, help="")

    args = parser.parse_args()

    set_seed(args.random_seed)
    train(args)
    
    
    
