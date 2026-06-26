import numpy
import os
import cv2
import torch
import numpy as np 
import tqdm
from torch.utils.data import Dataset
import json

from dataset.augmentations import *
from cfg import IN_mean, IN_std, ignore_datas, folded_3, folded_5

class HAI(Dataset):
    def __init__(
            self, 
            root_dir='./dataset/data/train', 
            det_path='./dataset/detection/processed/train_det.json', 
            det_crop=False,
            erase_other_car=False,
            mean=IN_mean, 
            std=IN_std, 
            fold_idx=0, 
            n_fold=5, 
            mode=0, 
            wh=(224, 224), 
            aug=True):
        
        self.root_dir = root_dir
        
        self.samples = []

        self.wh = wh 
        self.mode = mode
        self.aug = aug
        self.mean = mean
        self.std = std
        self.n_fold = n_fold
        if n_fold==3:
            folded = folded_3
        elif n_fold==5:
            folded = folded_5
        self.difficulty = 1.

        if erase_other_car:
            det_crop=True
        
        with open(det_path, mode='r') as f:
            self.det_data = json.load(f)

        self.det_crop = det_crop
        self.erase_other_car = erase_other_car

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
                    self.samples.append((img_path, fname, cls_idx))
        print("Ignored:", len(ignore_datas), ignore_datas)
        print("Filtered:", len(self.filtered), self.filtered)
        print("Class propotion", self.class_cnt)

    def __len__(self):
        return len(self.samples)

    def color_augmentation(self, image):
        # if np.random.uniform() < .5 * self.difficulty:
        #     image = rgb_shift(image, r_shift_limit=20, g_shift_limit=20, b_shift_limit=20)
        # if np.random.uniform() < .5 * self.difficulty:
        #     image = gaussian_noise(image, mean=0, std=1)
        if np.random.uniform() < .5* self.difficulty:
            image = color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        if np.random.uniform() < .5 * self.difficulty:
            image = image[..., np.random.choice([0, 1, 2], size=3, replace=False) ] # rgb shift
        return image

    def space_augmentation(self, image, mask=None):

        if np.random.uniform() < .1 * self.difficulty:
            image_tmp, mask_tmp = rotate(image, mask, np.random.uniform(-1, 1)*45*self.difficulty, value=(255, 255, 255))
            h, w, c = image_tmp.shape
            if (mask_tmp>0).sum() / (h*w*c) < 0.5:
                image = image_tmp
                mask = mask_tmp
        if np.random.uniform() < 0.25 * self.difficulty:
            image_tmp, mask_tmp = randomcrop(image, mask, min_ratio=.5)
            h, w, c = image_tmp.shape
            if (mask_tmp>0).sum() / (h*w*c) < 0.5:
                image = image_tmp
                mask = mask_tmp
        if np.random.uniform() < .5 * self.difficulty:
            image_tmp, mask_tmp = coarse_dropout(image, mask, num_holes_range=(1, 2), hole_height_range=(0.1, 0.3), hole_width_range=(0.1, 0.3), value=(255, 255, 255))
            h, w, c = image_tmp.shape
            if (mask_tmp>0).sum() / (h*w*c) < 0.5:
                image = image_tmp
                mask = mask_tmp
        if np.random.uniform() < .5 * self.difficulty:
            image = image[:, ::-1] # flip
            if mask is not None:
                mask = mask[:, ::-1] # flip
        return image, mask

    def augmentations(self, image, basename, vis=False):

        if self.det_crop or self.mode==1: # det_crop==True 인 경우와 아닌 경우의 Validaton이 달라지면 안된다고 판단
            det_x1, det_y1, det_x2, det_y2 = self.det_data[basename]['vehicle']
            image = image[det_y1:det_y2, det_x1:det_x2]

        mask = np.zeros_like(image)
        # white_mask = np.all(image == [255, 255, 255], axis=-1)
        # mask = np.zeros_like(image)
        # mask[white_mask] = image[white_mask]

        if self.erase_other_car and self.mode == 0 : # Train mode인 경우에만 적용하도록
            if np.random.uniform() < .2 * self.difficulty:
                for xyxy in self.det_data[basename]['else']['erase']:
                    x1, y1, x2, y2 = xyxy
                    y1 = max(y1-det_y1, 0)
                    x1 = max(x1-det_x1, 0)
                    y2 = y2-det_y1
                    x2 = x2-det_x1
                    image[y1:y2, x1:x2] = 255
                    mask[y1:y2, x1:x2] = 255 
                
        if self.aug:
            image, mask = self.space_augmentation(image, mask)
            image = self.color_augmentation(image)

        if vis==False:
            image = cv2.resize(image, self.wh)
            image = (image-self.mean)/self.std
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image).float()
        return image, mask
        
    def resample(self, idx):
        self.samples = [self.samples[i] for i in idx ]

    def __getitem__(self, idx, vis=False):
        img_path, basename, target = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, mask = self.augmentations(image, basename, vis=vis)

        return {
            'image': image,
            'target': target,
            'idx':idx,
            **({'mask':mask} if vis else {})
            }
