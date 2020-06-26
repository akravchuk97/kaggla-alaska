from efficientnet_pytorch import EfficientNet
from torch import nn


class EfficientNetAddCls(EfficientNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        x_qual = None
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.idx:
                x_qual = x
        x = self._swish(self._bn1(self._conv_head(x)))
        return x, x_qual

    def forward(self, inputs):
        bs = inputs.size(0)
        x, x_qual = self.extract_features(inputs)
        x_qual = self.avg_pool1(x_qual).view(bs, -1)
        x_qual = self.lin(x_qual)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x, x_qual
