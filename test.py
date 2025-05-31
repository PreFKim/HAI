import cv2
import numpy as np
import glob
import torch
import torch.nn.functional as F
import argparse
import pandas as pd
from cfg import mean, std
import tqdm
from model.baseline import BaseWraper
from dataset.augmentations import *
from functools import partial


def load_model(model_path, device='cpu'):
    w = torch.load(model_path, map_location=device)
    weight = w['model']
    model_config = w['config']
    class_names = w['class_names']

    print(f"Trained Epoch: {w['epoch']}")
    print("Model Config:")
    print(model_config)
    # print("Class names")
    # print(len(class_names), class_names)

    model = BaseWraper(**model_config).to(device).eval()

    print(model.load_state_dict(weight, strict=False))
    return model, class_names

def run(args, model_list, aug_list):
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    models = [load_model(path, device=device) for path in model_list ] # Ensemble은 K-Fold 로만 진행할 예정

    wh = models[0][0].wh 
    class_names = models[0][1]
    
    batch_size = len(images) if args.batch_size<=0 else args.batch_size
    filelist = sorted(glob.glob("./dataset/data/test/*.jpg"))
    results = []

    for i in tqdm.tqdm(range(0, len(filelist), batch_size), desc='Batch inferencing'):

        images = []
        for i, filename in enumerate(filelist[i:i+batch_size]):
            image = cv2.imread(filename)
            images.append([])
            for aug in aug_list: # Test-time Augmentation
                aug_image = aug(image.copy())
                aug_image = cv2.resize(aug_image, wh)
                aug_image = (aug_image-mean)/std
                aug_image = np.transpose(aug_image, (2, 0, 1))
                images[i].append(aug_image)

        images = np.array(images) # n_batch, n_aug, 3, h, w
        images = torch.from_numpy(images).float()

        b, n = images.shape[:2]
        with torch.no_grad():
            images = torch.flatten(images, 0, 1) # n_batch, n_aug, 3, h, w -> n_batch * n_aug, 3, h, w
            outputs = []
            for model, _ in models: # Ensemble
                output = model(images.to(device)) # n_batch * n_aug, n_class
                output = torch.unflatten(output, 0, (b, n)) # n_batch * n_aug, n_class - > n_batch, n_aug, n_class 
                outputs.append(output.cpu())
            outputs = torch.stack(outputs, dim=1) # [n_batch, n_aug, n_class] -> n_batch, n_models, n_aug, n_class 
            outputs = outputs.mean(dim=(1, 2)) # n_batch, n_models, n_aug, n_class -> n_batch, n_class 
            probs = F.softmax(outputs, dim=1) # n_batch, n_class

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
    submission.to_csv(f'output.csv', index=False, encoding='utf-8-sig')




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # 모델 파라미터
    parser.add_argument("--gpu", type=int, default=0, help="-1:CPU")
    parser.add_argument("--batch_size", type=int, default=128, help="0:All") 

    args = parser.parse_args()

    model_list = [
        "./experiments/14_0/best.pt",
        "./experiments/14_1/best.pt",
        "./experiments/14_2/best.pt",
        "./experiments/14_3/best.pt",
        "./experiments/14_4/best.pt",
    ]

    aug_list = [
        identity,
        flip,
        partial(scale, scale=0.5),
        partial(scale, scale=2.)
    ]

    run(args, model_list, aug_list)
    