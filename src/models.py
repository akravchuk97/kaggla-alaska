from efficientnet_pytorch import EfficientNet
from pretrainedmodels import se_resnext50_32x4d
from torch import nn

from effnet_cls import EfficientNetAddCls

_idx2n_units = {
    6: 48,
    8: 96,
    12: 96,
    14: 136,
    18: 232,
}


class Model(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.model = se_resnext50_32x4d()
        self.model.last_linear = nn.Linear(2048, n_classes)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.model(x)


def enetb2():
    net = EfficientNet.from_pretrained('efficientnet-b2')
    net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    return net


def enetb3():
    net = EfficientNet.from_pretrained('efficientnet-b3')
    net._fc = nn.Linear(in_features=1536, out_features=4, bias=True)
    return net


def enetb2_cls(idx=12):
    mdl = EfficientNetAddCls.from_pretrained('efficientnet-b2')
    mdl.idx = idx
    mdl._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
    mdl.lin = nn.Linear(120, 3)
    return mdl


def enetb3_cls(idx=12):
    mdl = EfficientNetAddCls.from_pretrained('efficientnet-b3')
    mdl.idx = idx
    mdl._fc = nn.Linear(in_features=1536, out_features=4, bias=True)
    mdl.lin = nn.Linear(_idx2n_units[idx], 3)
    return mdl
