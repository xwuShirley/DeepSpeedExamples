import torch
import torch.nn as nn


class CameraFuser(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.layer1 = nn.Linear(inchannels, outchannels)

    def forward(self, x):
        return self.layer1(x)