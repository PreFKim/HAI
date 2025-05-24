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

def run(args):
    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    exp_num, basename = args.checkpoint.split('/')[-2:]
    basename = basename.split(".")[0]

    print(exp_num)

    w = torch.load(args.checkpoint)
    weight = w['model']
    model_config = w['config']
    class_names = w['class_names']

    print(f"Trained Epoch: {w['epoch']}")
    print("Model Config:")
    print(model_config)
    print("Class names")
    print(len(class_names), class_names)

    model = BaseWraper(**model_config).to(device).eval()

    print(model.load_state_dict(weight, strict=False))
    
    images = []
    for filename in tqdm.tqdm(sorted(glob.glob("./dataset/data/test/*.jpg")), desc='Load images'):
        image = cv2.imread(filename)
        image = cv2.resize(image, model.wh)
        image = (image-mean)/std
        image = np.transpose(image, (2, 0, 1))
        images.append(image)

    images = np.array(images)
    images = torch.from_numpy(images).float()

    batch_size = len(images) if args.batch_size<=0 else args.batch_size

    results = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, len(images), batch_size), desc='Batch inferencing'):
            outputs = model(images[i:i+batch_size].to(device))
            probs = F.softmax(outputs, dim=1)

            # 각 배치의 확률을 리스트로 변환
            for prob in probs.cpu():  # prob: (num_classes,)
                result = {
                    class_names[i]: prob[i].item()
                    for i in range(len(class_names))
                }
                results.append(result)            
    pred = pd.DataFrame(results)
    submission = pd.read_csv('./dataset/data/sample_submission.csv', encoding='utf-8-sig')

    # 'ID' 컬럼을 제외한 클래스 컬럼 정렬
    class_columns = submission.columns[1:]
    pred = pred[class_columns]

    submission[class_columns] = pred.values
    submission.to_csv(f'./experiments/{exp_num}/{exp_num}_{basename}.csv', index=False, encoding='utf-8-sig')




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # 모델 파라미터
    parser.add_argument("--gpu", type=int, default=0, help="-1:CPU")
    parser.add_argument("--batch_size", type=int, default=128, help="0:All") 
    parser.add_argument("--checkpoint", type=str, default="./experiments/0/best.pt", help="")

    args = parser.parse_args()

    checkpoints = [
        ['./experiments/0/best.pt', 0.1],
    ]

    run(args)
    