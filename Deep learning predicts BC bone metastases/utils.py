import os
import sys
import json
import pickle
import random
import pandas as pd

import torch
from tqdm import tqdm
import sklearn.metrics as metrics

import matplotlib.pyplot as plt


def multi_models_roc(names, colors, y_test_S, y_predict_S, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上

    Args:
        names: list, 多个模型的名称
        sampling_methods: list, 多个模型的实例化对象
        save: 选择是否将结果保存（默认为png格式）

    Returns:
        返回图片对象plt
    """
    plt.figure(figsize=(30, 30), dpi=dpin)  # 画布
    plt.xticks(fontsize=40)  # 横坐标
    plt.yticks(fontsize=40)  # 纵坐标
    # plt.rc('font', **{'family': 'FZKaTong-M19S'})

    for (name, colorName, y_test, y_predict) in zip(names, colors, y_test_S, y_predict_S):
        # method.fit(X_train, y_train)
        # y_test_preds = method.predict(X_test)
        # y_test_predprob = method.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_true=y_test, y_score=y_predict, pos_label=1)

        plt.plot(fpr, tpr, lw=3, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)), color=colorName)
        plt.plot([0, 1], [0, 1], 'r--', lw=3, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('1-specificity', fontsize=40)
        plt.ylabel('Sensitivity', fontsize=40)
        plt.title('ROC comparision', fontsize=40)
        plt.legend(loc='lower right', fontsize=40)

    # if save:
    #     plt.savefig('multi_models_roc.png')

    return plt


def read_split_data(img_root: str, label_root: str, val_rate: float = 0.4):
    random.seed(3)  # 保证随机结果可复现
    """
    assert os.path.exists(img_root), "dataset root: {} does not exist.".format(img_root)
    
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    """
    label = pd.read_csv(label_root,usecols=['转移'],encoding='gb18030')
    label = label.values.tolist()
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    test_image_path = []
    test_image_label = []
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    images = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    """
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    """
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if os.path.splitext(file)[1] == '.npy':  # 判断，只记录npy
                images.append(os.path.join(root, file))
    val_path = random.sample(images, k=int(len(images) * val_rate))
    test_path = val_path[0:int(len(val_path)/2)]
    val_path = val_path[int(len(val_path)/2):]
    for img_path in images:
        i = img_path[6:]
        i = int(i[:-4])
        if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
            val_images_path.append(img_path)
            val_images_label.append(label[i-1])
        elif img_path in test_path:  # 否则存入训练集
            test_image_path.append(img_path)
            test_image_label.append(label[i-1])
        else:
            train_images_path.append(img_path)
            train_images_label.append(label[i-1])


    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    """
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()
    """

    return train_images_path, train_images_label, val_images_path, val_images_label,test_image_path,test_image_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1,1.4]).to(device))
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.squeeze().to(torch.int64)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        labels = labels.squeeze().to(torch.int64)
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
