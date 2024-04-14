import torch
import torch.nn as nn
from torch.autograd import Variable


class HandModel(nn.Module):
    def __init__(self, out_label_num,batch_size):
        super(HandModel, self).__init__()
        self.batch_size = batch_size
        self.con2d = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5)
        )

        self.flan = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(32 * 2 * 3, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.5),
            nn.Linear(16, out_label_num),
            nn.Softmax()
        )

    def forward(self, input):
        # (batch_size,
        input = input.permute(0, 2, 1)
        input = input.view(-1, 3, 6, 7)
        input = input.to(torch.float32)
        out = self.con2d(input)
        out = self.flan(out)
        out = self.linear(out)
        return out



if __name__ == "__main__":
    print(HandModel(out_label_num=4))
