from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
from os import listdir
import pandas as pd
import os
import cv2

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self,images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        #i = self.images_path[item][6:]
       # i = int(i[:-4])
        #img = np.load("/content/T11/Hog/" + str(i) + ".npy", allow_pickle=True).astype(float)
        img = np.load(self.images_path[item], allow_pickle=True).astype(float)
        img = torch.tensor(img, dtype=torch.float32)
        img = img.unsqueeze(0)
        # img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        """
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        """
        label = self.images_class[item]
       # if (label.item() == 0):
       #     label = torch.tensor(np.array([0,1]))
       # elif (label.item() == 1):
       #     label = torch.tensor(np.array([1, 0]))

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

