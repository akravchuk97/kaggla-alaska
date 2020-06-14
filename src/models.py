import os

from torch import nn
from pretrainedmodels import se_resnext50_32x4d

os.environ['TORCH_HOME'] = '/home/ds'


class Model(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = se_resnext50_32x4d()
        self.model.last_linear = nn.Linear(2048, n_classes)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.model(x)


def get_model(n_classes=4):
    return Model(n_classes=n_classes)
