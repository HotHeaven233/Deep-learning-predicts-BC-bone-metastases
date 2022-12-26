import os
from ResNetmodel import *
#from Result import *
#from xception import *
#from densenet import *
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from my_dataset import MyDataSet
from utils import read_split_data#,load_from_pretrained
import torch.optim as optim
from tqdm import tqdm
from torch.utils import data
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

root = './T11'
feature_root = './BC-Data.csv'

def main():
    data = pd.read_csv(feature_root,encoding='gb18030')
    #data = data.iloc[:, 1:19]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    name = 'IMG3'

    train_images_path = np.load('./split_data/'+str(name)+'/train_images_path.npy')
    train_images_path = train_images_path.tolist()
    train_images_label = np.load('./split_data/'+str(name)+'/train_images_label.npy').tolist()
    val_images_path = np.load('./split_data/'+str(name)+'/val_images_path.npy').tolist()
    val_images_label = np.load('./split_data/'+str(name)+'/val_images_label.npy').tolist()

    #train_images_label, val_images_path, val_images_label,test_image_path,test_image_label= read_split_data(root, feature_root)
    train_images_label = torch.tensor(train_images_label)
    train_images_label = train_images_label.float()
    val_images_label = torch.tensor(val_images_label)
    val_images_label = val_images_label.float()
    data_transform = {
        "train": transforms.Compose([#transforms.CenterCrop(size=(400,1000)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     #transforms.ToTensor(),
                                     transforms.Normalize([0.5], [0.5])
                                     ]),
        "val": transforms.Compose([#transforms.CenterCrop(size=(400,1000)),
                                   #transforms.Resize(256),
                                   #transforms.CenterCrop(224),
                                   #transforms.ToTensor(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.Normalize([0.5], [0.5])
                                   ])}

    train_data_set = MyDataSet(
                               images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform['train']
                               )
    val_data_set = MyDataSet(
                             images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform['val']
                             )

    batch_size = 2
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #nw = 0
    print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nw,
                                               collate_fn=None)
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=nw,
                                             collate_fn=None)
    train_num = len(train_data_set)
    val_num = len(val_data_set)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = resnet34()
    #pretext_model = torch.load('densenet121-a639ec97.pth')
    #model2_dict = net.state_dict()
    #state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
    #model2_dict.update(state_dict)
    #net.load_state_dict(model2_dict)

    net.to(device)
    #net = xception()


    loss_function = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0,1.0]).to(device))

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-6)#xception 0.000005,resnet34 0.000001,resnet50 0.000003
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    sigmoid = nn.Sigmoid()

    epochs = 300
    best_acc = 0
    train_steps = len(train_loader)
    val_steps = len(val_loader)

    writer = SummaryWriter('./Result/tb_densenet121')

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        test_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, (imgs, labs) in enumerate(train_bar):
            images = imgs
            labels = labs
            optimizer.zero_grad()
            logits = net(images.to(device))

            labels = labels.squeeze(1).to(torch.int64)
            pred_classes = torch.max(logits, dim=1)[1]

            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        scheduler.step()
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            for imgs, labs in val_bar:
                val_images, val_labels = imgs, labs
                val_labels = val_labels.squeeze().to(torch.int64)
                outputs = net(val_images.to(device))
                val_loss = loss_function(outputs, val_labels.to(device))
                outputs = torch.max(outputs, dim=1)[1]

                acc += torch.eq(outputs, val_labels.to(device)).sum().item()

                test_loss += val_loss.item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_loss: %.3f accurate: %.3f'%
              (epoch + 1, running_loss / train_steps, test_loss / val_steps,val_accurate ))
        ValLoss = test_loss / val_steps

        writer.add_scalar("TrainLoss", running_loss / train_steps, epoch)
        writer.add_scalar("ValLoss", test_loss / val_steps, epoch)
        writer.add_scalar("acc", val_accurate, epoch)

        torch.save(net.state_dict(), "./weights/model-{}.pth".format(epoch+1))

    print('Finished Training best loss:')
    print(best_acc)


if __name__ == '__main__':
    main()
