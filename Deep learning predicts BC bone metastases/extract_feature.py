import os
import json
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import csv
from ResNetmodel import *
from Result import *
import pandas as pd
#from model import resnet50


activation1 = {}
def get_activation1(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation1[name] = output.detach()
    return hook

activation2 = {}
def get_activation2(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation2[name] = output.detach()
    return hook

activation3 = {}
def get_activation3(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation3[name] = output.detach()
    return hook
def main():

    df = pd.read_csv('./breast.csv', encoding='gb18030')
    #data = df[df['序号'].isin([1])]
    data = df[df.columns[0:29]]
    label = df[['序号','骨转移']]
    #a = data.iloc[0, 11]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [#transforms.CenterCrop(size=(400,1000)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         #transforms.Resize(256),
         #transforms.CenterCrop(224),
         #transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
         ])

    train_images_path1 = np.load('./split_data/LBP3/train_images_path.npy').tolist()
    train_images_label1 = np.load('./split_data/LBP3/train_images_label.npy').tolist()
    val_images_path1 = np.load('./split_data/LBP3/val_images_path.npy').tolist()
    val_images_label1 = np.load('./split_data/LBP3/val_images_label.npy').tolist()
    test_images_path1 = np.load('./split_data/LBP3/test_images_path.npy').tolist()
    test_images_label1 = np.load('./split_data/LBP3/test_images_label.npy').tolist()

    train_images_path2 = np.load('./split_data/Hog3/train_images_path.npy').tolist()
    train_images_label2 = np.load('./split_data/Hog3/train_images_label.npy').tolist()
    val_images_path2 = np.load('./split_data/Hog3/val_images_path.npy').tolist()
    val_images_label2 = np.load('./split_data/Hog3/val_images_label.npy').tolist()
    test_images_path2 = np.load('./split_data/Hog3/test_images_path.npy').tolist()
    test_images_label2 = np.load('./split_data/Hog3/test_images_label.npy').tolist()

    train_images_path3 = np.load('./split_data/IMG3/train_images_path.npy').tolist()
    train_images_label3 = np.load('./split_data/IMG3/train_images_label.npy').tolist()
    val_images_path3 = np.load('./split_data/IMG3/val_images_path.npy').tolist()
    val_images_label3 = np.load('./split_data/IMG3/val_images_label.npy').tolist()
    test_images_path3 = np.load('./split_data/IMG3/test_images_path.npy').tolist()
    test_images_label3 = np.load('./split_data/IMG3/test_images_label.npy').tolist()

    model1 = resnet101().to(device)
    # load model weights
    model1_weight_path = "./weights/LBP3-Resnet101/model-300.pth"
    model1.load_state_dict(torch.load(model1_weight_path, map_location=device))

    model2 = resnet101().to(device)
    # load model weights
    model2_weight_path = "./weights/Hog3-Resnet101/model-300.pth"
    model2.load_state_dict(torch.load(model2_weight_path, map_location=device))

    model3 = resnet101().to(device)
    # load model weights
    model3_weight_path = "./weights/IMG3-Resnet101/model-299.pth"
    model3.load_state_dict(torch.load(model3_weight_path, map_location=device))

    for img_path1,img_path2,img_path3 in zip(val_images_path1,val_images_path2,val_images_path3):  #
        list = []

        img1 = np.load(img_path1, allow_pickle=True).astype(float)
        img1 = torch.tensor(img1, dtype=torch.float32)
        img1 = torch.unsqueeze(img1, dim=0)
        img1 = data_transform(img1)
        img1 = torch.unsqueeze(img1, dim=0)

        img2 = np.load(img_path2, allow_pickle=True).astype(float)
        img2 = torch.tensor(img2, dtype=torch.float32)
        img2 = torch.unsqueeze(img2, dim=0)
        img2 = data_transform(img2)
        img2 = torch.unsqueeze(img2, dim=0)

        img3 = np.load(img_path3, allow_pickle=True).astype(float)
        img3 = torch.tensor(img3, dtype=torch.float32)
        img3 = torch.unsqueeze(img3, dim=0)
        img3 = data_transform(img3)
        img3 = torch.unsqueeze(img3, dim=0)

        model1.eval()
        with torch.no_grad():
            model1.avgpool.register_forward_hook(get_activation1('1'))
            output = torch.squeeze(model1(img1.to(device))).cpu()
            i = img_path1[6:]
            i = int(i[:-4])  #
            list.append(i)

            tensor = activation1['1'].cpu()  # .tolist()
            tensor = torch.squeeze(tensor)
            t = tensor[0].float().item()

            for x in range(0, len(tensor)):
                list.append(tensor[x].float().item())

        model2.eval()
        with torch.no_grad():
            model2.avgpool.register_forward_hook(get_activation2('2'))
            output = torch.squeeze(model2(img2.to(device))).cpu()
            i = img_path2[6:]
            i = int(i[:-4])  #
            #list.append(i)

            tensor = activation2['2'].cpu()  # .tolist()
            tensor = torch.squeeze(tensor)
            t = tensor[0].float().item()

            for x in range(0, len(tensor)):
                list.append(tensor[x].float().item())

        model3.eval()
        with torch.no_grad():
            model3.avgpool.register_forward_hook(get_activation3('3'))
            output = torch.squeeze(model3(img3.to(device))).cpu()
            i = img_path3[6:]
            i = int(i[:-4])  #
            #list.append(i)

            tensor = activation3['3'].cpu()  # .tolist()
            #print(tensor)
            tensor = torch.squeeze(tensor)
            t = tensor[0].float().item()

            for x in range(0, len(tensor)):
                list.append(tensor[x].float().item())
            struct_data = data[data['序号'].isin([str(i)])]
            struct_label = label[label['序号'].isin([str(i)])]
            del struct_data['序号']
            del struct_label['序号']
            #print(struct_data[1,11])
            for x in range(0, struct_data.shape[1]):
                list.append(struct_data.iloc[0,x])
            list.append(struct_label.iloc[0,0])

        with open("./Result/T11-val.csv", "a", newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(list)

if __name__ == '__main__':
    main()
