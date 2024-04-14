import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from Handmodel import HandModel
from get_data_path import get_data_path
import os
import random
# from torchnet import meter
from torch.autograd import Variable
import copy
rootFpath = 'E:/Demo_Project/SignLanguageRecognition/applacation/npz_files/'  # 文件根目录

out_label_num = len(os.listdir(rootFpath))

batch_size = 128

class dataset(Dataset):
    def __init__(self,data, label):
        # 遍历npz取数据
        self.data = data
        self.label = label

    def test(self):
        return self.data, self.label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.data[item], self.label[item]


path = get_data_path(rootFpath, Access='train')
data = []
label = []
for index, p in enumerate(path):
    l, fpath = p[0], p[1]
    da = np.load(fpath)
    da = da['data']
    data.extend(da)
    for i in range(len(da)):
        label.append(index)

data = np.array(data)
label = np.array(label)

splt = len(label)
random_idx = random.sample(range(0, splt), splt)
train_idx = random_idx[0:int(0.8*splt)]
test_idx = random_idx[int(0.8*splt)+1:splt]

train_data = data[train_idx]
train_label = label[train_idx]
test_data = data[test_idx]
test_label = label[test_idx]


myTrainDataSet = dataset(train_data,train_label)
myTestDataSet = dataset(test_data,test_label)

# 定义自己的dataloader
train_loader = DataLoader(dataset=myTrainDataSet, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=myTestDataSet, batch_size=batch_size, shuffle=True)

model = HandModel(out_label_num=out_label_num,batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# model.load_state_dict(torch.load("11_16_param.pkl"))

# 选择损失函数和优化方法
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(200):
    model.train()
    for step, (sensor, label) in enumerate(train_loader):

        sensor = sensor.float()
        sensor = sensor.to(device)
        label = label.long()
        label = label.to(device)

        logits = model(sensor)


        loss = loss_func(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每100个step，输出一下训练的情况
        if step % 100 == 0:
            print("train epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}".
                  format(epoch,
                         step * len(sensor),
                         len(train_loader.dataset),
                         100. * step / len(train_loader),
                         loss.item()))

    # 完成一个epoch，看一下loss
    print("\t===============epoch {} done, the loss is {:.5f}===============\t".format(epoch, loss.item()))
    # 完成一个epoch，画一下图
    # viz.line([loss.item()], [epoch], win="train_loss", update="append")

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for step, (sensor, label) in enumerate(test_loader):
            sensor = sensor.float()
            sensor = sensor.to(device)
            label = label.long()
            label = label.to(device)

            logits = model(sensor)

            pred = logits.argmax(dim=1)

            # print(pred)
            # print(label)
            # tensor([5, 5, 5, 5, 0, 5, 5, 5], device='cuda:0')
            # tensor([2, 4, 4, 4, 3, 2, 4, 5], device='cuda:0')

            total_correct += torch.eq(pred, label).float().sum()
            total_num += sensor.size(0)

        acc = total_correct / total_num

    # 看一下正确率
    print("\t===============epoch {} done, the ACC is {:.5f}===============\t".format(epoch, acc.item()))
    # viz.line([acc.item()], [epoch], win="test_acc", update="append")
    best = 0
    if (acc.item() >= best):
        torch.save(model.state_dict(), "11_16_param.pkl")
        best = acc.item()

