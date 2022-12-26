import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

for n in range(613,614):
    num = n
    #img = cv2.imread('./data/Cropped_img/'+str(num)+'-1.png', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(r"./img/"+str(num)+".jpg", 0)        # 读取图片，有0的话表示转变为灰度图；
    image = np.float32(image) / 255.0       # 归一化
    # cv2.imshow("normalization",image)
    # cv2.waitKey()

    fd, hog_image = hog(image,
                      orientations=8,
                      pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1),
                      visualize=True,
                      multichannel=False) # multichannel=True是针对3通道彩色；

    np.save(file='./Hog/' + str(num) + '.npy', arr=hog_image)
