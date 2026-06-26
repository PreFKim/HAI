import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
import tqdm
from dataset.augmentations import *
from functools import partial
from utils import load_model
import itertools
import json
import os

same_class_names = [
    ['K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'],
    ['디_올뉴니로_2022_2025', '디_올_뉴_니로_2022_2025'],
    ['718_박스터_2017_2024', '박스터_718_2017_2024'],
    ['RAV4_2016_2018', '라브4_4세대_2013_2018'],
    ['RAV4_5세대_2019_2024', '라브4_5세대_2019_2024'],
]

def run(args, model_list, aug_list):
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    cache_name = f'cache{"_erase" if args.erase_other_car else ""}.pt'
    models = [] # Ensemble은 K-Fold 로만 진행할 예정
    caches = []
    for path in model_list :
        cache_path = os.path.join(os.path.dirname(path), cache_name)
        exp_name = cache_path.split("/")[-1]
        if os.path.exists(cache_path) and args.use_cache:
            print(cache_path)
            models.append(load_model(path, device='cpu', ema=args.use_ema))
            caches.append(torch.load(cache_path))
            print(f"{exp_name} has cache file {cache_path}")
        else:
            models.append(load_model(path, device=device, ema=args.use_ema))
            caches.append(None)
        
    combinations = list(itertools.product(*aug_list))
    wh = [int(s*args.resolution_multiplier) for s in models[0][0].wh ]
    print("wh:",wh)
    class_names = models[0][1]
    mean = torch.tensor(models[0][0].mean).to(device)
    std = torch.tensor(models[0][0].std).to(device)
    
    filelist = sorted(glob.glob("./dataset/data/test/*.jpg"))
    batch_size = len(filelist) if args.batch_size<=0 else args.batch_size
    results = []

    if args.apply_same_class :
        same_class_idxs = []

        for i, pair in enumerate(same_class_names):
            same_class_idxs.append([])
            for cls_name in pair:
                same_class_idxs[i].append(class_names.index(cls_name))
        print(same_class_names)    
        print(same_class_idxs)


    if args.erase_other_car:
        args.det_crop = True

    if args.det_crop:
        with open("./dataset/detection/processed/test_det.json", mode='r') as f:
            test_det_crop = json.load(f)
    saved_outputs=[]
    for i in tqdm.tqdm(range(0, len(filelist), batch_size), desc='Batch inferencing'):

        if None in caches:
            images = []
            for j, filename in enumerate(filelist[i:i+batch_size]):
                basename = filename.split("/")[-1]
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if args.erase_other_car : 
                    for xyxy in test_det_crop[basename]['else']['erase']:
                        x1, y1, x2, y2 = xyxy
                        image[y1:y2, x1:x2] = 255

                if args.det_crop:
                    det_x1, det_y1, det_x2, det_y2 = test_det_crop[basename]['vehicle']
                    image = image[det_y1:det_y2, det_x1:det_x2]

                images.append([])
                for augs in combinations: # Test-time Augmentation
                    aug_image = image.copy()
                    for aug in augs:
                        aug_image = aug(aug_image)
                    aug_image = cv2.resize(aug_image, wh)
                    images[j].append(aug_image)
            images = np.array(images, dtype=np.uint8) # n_batch, n_aug, 3, h, w
            images = torch.from_numpy(images).float().to(device)
            with torch.no_grad():
                images = (images-mean)/std
                images = torch.permute(images, (0, 1, 4, 2, 3))
                images = torch.flatten(images, 0, 1) # n_batch, n_aug, 3, h, w -> n_batch * n_aug, 3, h, w

        with torch.no_grad():
            outputs = []
            for j, (model, _) in enumerate(models): # Ensemble
                if caches[j] is not None and args.use_cache:
                    output = caches[j][i:i+batch_size]
                else:
                    model = model.eval()
                    output = model(images)['output'] # n_batch * n_aug, n_class
                    output = torch.unflatten(output, 0, (-1, len(combinations))).cpu() # n_batch * n_aug, n_class - > n_batch, n_aug, n_class 
                outputs.append(output)
            outputs = torch.stack(outputs, dim=1) # [n_batch, n_aug, n_class] -> n_batch, n_models, n_aug, n_class 
            if args.use_cache:
                saved_outputs.append(outputs)
            outputs = outputs.mean(dim=(1, 2)) # n_batch, n_models, n_aug, n_class -> n_batch, n_class 

            probs = F.softmax(outputs/args.temperature, dim=1) # n_batch, n_class

            if args.apply_same_class :
                conf, cls_idx = torch.max(probs, dim=-1) # 최종 클래스 선택
                probs_modified = probs.clone()
                for pair in same_class_idxs: # 예측한 클래스가 동일 클래스를 가진 경우 = Sum, 아니면 최소값 대체
                    a, b = pair

                    mask = ((cls_idx == a) | (cls_idx == b))  # n_batch
                    sum_vals = probs[:, a] + probs[:, b]
                    min_vals = torch.min(probs[:, a], probs[:, b])

                    probs_modified[:, a] = torch.where(mask, sum_vals, min_vals)
                    probs_modified[:, b] = torch.where(mask, sum_vals, min_vals)
                    # print(cls_idx, pair, probs_modified[:, a], probs_modified[:, b])
                
                probs = probs_modified
            
            # 각 배치의 확률을 리스트로 변환
            for prob in probs.cpu():  # prob: (num_classes,)
                result = {
                    class_names[j]: prob[j].item() # 모델별로 클래스가 다르다면 torch.stack 부분에서 막힘
                    for j in range(len(class_names))
                }
                results.append(result)            
    pred = pd.DataFrame(results)
    submission = pd.read_csv('./dataset/data/sample_submission.csv', encoding='utf-8-sig')

    # 'ID' 컬럼을 제외한 클래스 컬럼 정렬
    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(args.output_path, index=False, encoding='utf-8-sig')
    print(f"Saved at {args.output_path}")

    if args.use_cache:
        saved_outputs = torch.cat(saved_outputs, 0) # 8258, n_model, n_aug, n_class
        for i, path in enumerate(model_list):
            torch.save(saved_outputs[:, i], os.path.join(os.path.dirname(path), cache_name))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # 모델 파라미터
    parser.add_argument("--gpu", type=int, default=0, help="-1:CPU")
    parser.add_argument("--use_ema", action='store_true', help="")
    parser.add_argument("--det_crop", action='store_true', help="") # 필수
    parser.add_argument("--erase_other_car", action='store_true', help="")
    parser.add_argument("--use_cache", action='store_true', help="") # 필수
    parser.add_argument("--apply_same_class", action='store_true', help="")
    parser.add_argument("--resolution_multiplier", type=float, default=1., help="") 
    parser.add_argument("--temperature", type=float, default=1., help="High -> close uniform,") 
    parser.add_argument("--batch_size", type=int, default=1, help="0:All") 
    parser.add_argument("--fold_type", type=str, default='b', help="") 
    parser.add_argument("--output_path", type=str, default='./output.csv', help="") 

    args = parser.parse_args()

    args.output_path = f'{args.output_path.split(".")[0]}{"_det"if args.det_crop else ""}{"_erase"if args.erase_other_car else ""}.{args.output_path.split(".")[1]}'

    if args.fold_type == 'b':
        model_list = [
            "./experiments/pretrain_0/best.pt",
            "./experiments/pretrain_1/best.pt",
            "./experiments/pretrain_2/best.pt",
            "./experiments/pretrain_3/best.pt",
            "./experiments/pretrain_4/best.pt",
        ]
    elif args.fold_type == 's':
        model_list = [
            "./experiments/pretrain3_0/best.pt",
            "./experiments/pretrain3_1/best.pt",
            "./experiments/pretrain3_2/best.pt",
            "./experiments/pretrain3_3/best.pt",
            "./experiments/pretrain3_4/best.pt",
        ]
    else:
        model_list = [
            "./experiments/pretrain_0/best.pt",
            "./experiments/pretrain_1/best.pt",
            "./experiments/pretrain_2/best.pt",
            "./experiments/pretrain_3/best.pt",
            "./experiments/pretrain_4/best.pt",
            "./experiments/pretrain3_0/best.pt",
            "./experiments/pretrain3_1/best.pt",
            "./experiments/pretrain3_2/best.pt",
            "./experiments/pretrain3_3/best.pt",
            "./experiments/pretrain3_4/best.pt",
        ]

    aug_list = [
        [identity, flip],
        [
            partial(scale, scale=0.75, value=255),
            identity,
            partial(scale, scale=1.25, value=255),
            partial(scale, scale=1.5, value=255),
            partial(scale, scale=2.0, value=255),
        ]
    ] 

    run(args, model_list, aug_list)
    