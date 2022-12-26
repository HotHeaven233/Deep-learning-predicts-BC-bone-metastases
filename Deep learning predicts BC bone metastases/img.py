import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import os

images = []
img_root = r'./img'
for root, dirs, files in os.walk(img_root):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':  # 判断，只记录npy
            images.append(os.path.join(root, file))

for img in images:
    num = img[6:]
    num = int(num[:-4])
    image = cv2.imread(img, 0)
    np.save(file='./T11/' + str(num) + '.npy', arr=image)