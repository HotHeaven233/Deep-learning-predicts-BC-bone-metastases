from skimage.feature import local_binary_pattern
from skimage import data, filters
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# settings for LBP
radius = 3  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数
images = []
img_root = r'./img'
for root, dirs, files in os.walk(img_root):
    for file in files:
        if os.path.splitext(file)[1] == '.jpg':  # 判断，只记录npy
            images.append(os.path.join(root, file))

for img in images:
    num = img[6:]
    num = int(num[:-4])

    # 读取图像
    image = cv2.imread(r"./img/"+str(num)+".jpg",cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    """
    #显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.subplot(221)   # 把画布分成2*2的格子，放在第一格
    plt.imshow(image1)

    # 转换为灰度图显示
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(222)  # 把画布分成2*2的格子,放在第2格
    plt.imshow(image, cmap='gray')
    """
    # LBP特征
    lbp = local_binary_pattern(image, n_points, radius)
    np.save(file='./LBP/' + str(num) + '.npy', arr=lbp)
"""
# 边缘特征-sobel算子
edges = filters.sobel(image)
plt.subplot(224)  # 把画布分成2*2的格子,放在第4格
plt.imshow(edges, cmap='gray')
plt.show()
"""
#plt.show()
