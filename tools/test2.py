
import numpy as np
import mindspore
from mindspore import nn, ops
from mindspore import Tensor, Parameter

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 5, pad_mode="valid")
        self.param = Parameter(Tensor(np.array([1.0], np.float32)), 'param')

    def construct(self, x):
        x = self.conv(x)
        x = x * self.param
        out = ops.matmul(x, x)
        return out

net = Net()

# 配置优化器需要更新的参数
optim = nn.Adam(params=net.trainable_params())
print(net.trainable_params())
for name, param in net.parameters_and_names():
    print(f"Layer: {name}\nSize: {param.shape}\nValues : {param[:2]} \n")