import os
import json
import csv
import torch
from PIL import Image
from torchvision import transforms
from ResNetmodel import *
import matplotlib.pyplot as plt
from utils import read_split_data, multi_models_roc
import numpy as np
import sklearn as skl
from sklearn.metrics import accuracy_score
#from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [#transforms.Resize(256),
         #transforms.CenterCrop(224),
         #transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.Normalize([0.5], [0.5])
        ])

    # load image
    #data_path = "./LBP"
    #label_root = './BC-Data.csv'
    name = 'LBP3'
    #assert os.path.exists(data_path), "file: '{}' dose not exist.".format(data_path)
    train_images_path = np.load('./split_data/'+str(name)+'/train_images_path.npy')
    train_images_path = train_images_path.tolist()
    train_images_label = np.load('./split_data/'+str(name)+'/train_images_label.npy').tolist()
    val_images_path = np.load('./split_data/'+str(name)+'/val_images_path.npy').tolist()
    val_images_label = np.load('./split_data/'+str(name)+'/val_images_label.npy').tolist()
    test_images_path = np.load('./split_data/'+str(name)+'/test_images_path.npy').tolist()
    test_images_label = np.load('./split_data/'+str(name)+'/test_images_label.npy').tolist()

    model = resnet101().to(device)
    # load model weights
    model_weight_path = "./weights/LBP3-Resnet101/model-300.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    predict_pro = []
    predict_cla = []
    y_test_S = [test_images_label]  #

    for img_path in test_images_path:  #
        img = np.load(img_path, allow_pickle=True).astype(float)
        img = torch.tensor(img, dtype=torch.float32)
        img = torch.unsqueeze(img, dim=0)
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)


        model.eval()
        with torch.no_grad():
           # predict class
            a = model(img.to(device))
            #print(a)
            output = torch.squeeze(model(img.to(device))).cpu()
            #print(output)
            predict = torch.softmax(output, dim=0)
            output = predict[1]
            threshold = 0.16
            output = torch.where(output >= threshold, torch.ones_like(output), output)
            output = torch.where(output < threshold, torch.zeros_like(output), output)
            predict_class = output.numpy()


            predict_probability = predict.numpy()
            # predict_probability = predict_probability[1]
            predict_pro.append(predict_probability[1].item())

            predict_cla.append(predict_class.item())

    names = ["ROC"]
    colors = ['black']
    y_predict_S = [predict_pro]
    #a = np.array(test_images_label)
    #b = np.array(predict_pro)
    #np.save('cla_vit.npy', a)
    #np.save('pro_vit.npy', b)
    list = []
    #train_roc_graph = multi_models_roc(names, colors, y_test_S, y_predict_S)
    matrix = skl.metrics.confusion_matrix(test_images_label, predict_cla, labels=[0, 1])  #
    tn, fp, fn, tp = matrix.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    acc = accuracy_score(test_images_label, predict_cla)  #
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    #print(train_images_label)
    #print(predict_cla)
    auc = skl.metrics.roc_auc_score(test_images_label,predict_pro)  #

    print('AUC:', auc)
    print('acc: ', acc)
    print('sensitivity: ', sen)
    print('specificity: ', spe)
    print('ppv:', ppv)
    print('npv:', npv)
    list.append(auc)
    list.append(acc)
    list.append(sen)
    list.append(spe)
    list.append(ppv)
    list.append(npv)

    with open("./Result/LBP3-resnet101-test.csv", "a", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(list)

    plt.show()


if __name__ == '__main__':
    #for i in range(0,50):
       main()
