import numpy
import os
import cv2
import torch
import numpy as np 
from torch.utils.data import Dataset

from cfg import mean, std

class HAI(Dataset):
    def __init__(self, root_dir='./dataset/data/train', image_size=(224, 224), aug=True):
        self.root_dir = root_dir
        
        self.samples = []

        self.image_size = image_size # (w, h)

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith(('.jpg')):
                    img_path = os.path.join(cls_folder, fname)
                    target = self.class_to_idx[cls_name]
                    self.samples.append((img_path, target))

    def __len__(self):
        return len(self.samples)
    
    def resample(self, idx):
        self.samples = [self.samples[i] for i in idx ]

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.resize(image, self.image_size)
        image = (image-mean)/std
        image = np.transpose(image, (2, 0, 1))

        image = torch.from_numpy(image).float()

        return {
            'image': image,
            'target': target,
            'idx':idx
            }
