#导入训练模型需要的库、包
import numpy as np
import torch as t
from model_m import HandModel_
from mindspore import nn
from torchnet import meter
from torch.autograd import Variable
import copy

label=["you","two","eight", "no", "hit", "ugly", "praise", "one", "like", "study", "three",
         "moved",
         "",
         "", "", "", "", "", "", "", "", "", "", "", "",
         "",
         ""]
#two three study you不能用

label_num = len(label)#提取出label列的长度
# 模型保存地址即是label+.npz

targetX = [0 for xx in range(label_num)]
target = []
for xx in range(label_num):
    target_this = copy.deepcopy(targetX)
    target_this[xx] = 1
    target.append(target_this)
# 独热码
lr = 1e-3  # learning rate
model_saved = 'checkpoints/model'

# 模型定义
model = HandModel_()
# optimizer = t.optim.Adam(model.parameters(), lr=lr)
optim = nn.Adam(params=model.trainable_params(), learning_rate=lr)
criterion = nn.CrossEntropyLoss()

loss_meter = meter.AverageValueMeter()

epochs = 40
for epoch in range(epochs):
    print("epoch:" + str(epoch))
    loss_meter.reset()
    count = 0.1
    allnum = 0.1
    for i in range(len(label)):
        data = np.load('./npz_files/' + label[i] + ".npz", allow_pickle=True)
        data = data['data']

        for j in range(len(data)):
            xdata = t.tensor(data[j])
            optimizer.zero_grad()
            this_target = t.tensor(target[i]).float()
            input_, this_target = Variable(xdata), Variable(this_target)

            output = model(input_)

            outLabel = label[output.tolist().index(max(output))]
            targetIndex = target[i].index(1)
            targetLabel = label[targetIndex]
            if targetLabel == outLabel:
                count += 1
            allnum += 1

            output = t.unsqueeze(output, 0)
            this_target = t.unsqueeze(this_target, 0)

            loss = criterion(output, this_target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.data)

    print("correct_rate:", str(count / allnum))

    t.save(model.state_dict(), '%s_%s.pth' % (model_saved, epoch))
