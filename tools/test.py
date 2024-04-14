from mindspore.nn import Cell, Dense, MSELoss
import mindspore
from mindspore import Tensor, load_checkpoint
from mindspore import dtype
import numpy as np


class HandModel_(Cell):
    def __init__(self):
        super(HandModel_, self).__init__()
        self.linear1 = Dense(in_channels=48, out_channels=40)
        self.linear2 = Dense(in_channels=40, out_channels=32)
        self.linear3 = Dense(in_channels=32, out_channels=27)

    def construct(self, input):
        out = self.linear1(input)
        out = self.linear2(out)
        out = self.linear3(out)

        return out


# 示例代码
model = HandModel_()
x = Tensor(np.random.randn(10, 48).astype(np.float32))
y = Tensor(np.random.randn(10, 27).astype(np.float32))
optimizer = mindspore.nn.optim.Adam(params=model.trainable_params(), learning_rate=0.01)
criterion = MSELoss()
for i in range(100):
    output = model(x)
    loss = criterion(output, y)
    optimizer.backward(loss)
    optimizer.step()
    print('step: {}, loss: {}'.format(i + 1, loss.asnumpy()))
