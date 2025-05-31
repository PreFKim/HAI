import numpy
import os
import cv2
import torch
import numpy as np 
import tqdm
from torch.utils.data import Dataset

from dataset.augmentations import *
from cfg import mean, std, ignore_datas, folded

class HAI(Dataset):
    def __init__(self, root_dir='./dataset/data/train', fold_idx=0, mode=0, wh=(224, 224), aug=True):
        self.root_dir = root_dir
        
        self.samples = []

        self.wh = wh 
        self.mode = mode
        self.aug = aug

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.class_cnt = [0 for _ in range(len(self.classes))]

        self.filtered = []

        for cls_idx, cls_name in enumerate(tqdm.tqdm(self.classes, desc='Processing')):
            cls_folder = os.path.join(root_dir, cls_name)
            for data_idx, fname in enumerate(sorted(os.listdir(cls_folder))):
                if fname in ignore_datas:
                    self.filtered.append(fname)    
                elif (fold_idx >= 0 and (
                    data_idx in folded[cls_idx][fold_idx] and mode==0 or 
                    data_idx not in folded[cls_idx][fold_idx] and mode==1)):
                    pass
                elif fname.lower().endswith(('.jpg')):
                    self.class_cnt[cls_idx] += 1
                    img_path = os.path.join(cls_folder, fname)
                    self.samples.append((img_path, cls_idx))
        print("Ignored:", len(ignore_datas), ignore_datas)
        print("Filtered:", len(self.filtered), self.filtered)
        print("Class propotion", self.class_cnt)

    def __len__(self):
        return len(self.samples)

    def augmentations(self, image):
        if self.aug:
            if np.random.uniform() < .5:
                image = coarse_dropout(image, num_holes_range=(1, 2), hole_height_range=(0.1, 0.3), hole_width_range=(0.1, 0.3), value=(255, 255, 255))
            if np.random.uniform() < .5:
                image = color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            if np.random.uniform() < .5:
                image = image[:, ::-1] # flip
            if np.random.uniform() < .5:
                image = image[..., np.random.choice([0, 1, 2], size=3, replace=False) ] # rgb shift
            # if np.random.uniform() < .1:
            #     image = rotate(image, np.random.uniform(-1, 1)*180)
            if np.random.uniform() < .25:
                image = randomcrop(image, min_ratio=.5)

        image = cv2.resize(image, self.wh)
        image = (image-mean)/std
        image = np.transpose(image, (2, 0, 1))
        return image
        
    def resample(self, idx):
        self.samples = [self.samples[i] for i in idx ]

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = cv2.imread(img_path)
        image = self.augmentations(image)

        image = torch.from_numpy(image).float()

        return {
            'image': image,
            'target': target,
            'idx':idx
            }
