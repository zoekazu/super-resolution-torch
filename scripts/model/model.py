"""Model meta definition


<explanation>
"""


import abc

from torch import nn


class MetaNet(nn.Module, abc.ABC):
    def __init__(self, upscale_factor, input_ch=1):
        super(MetaNet, self).__init__()
        self._upscale_factor = upscale_factor
        self._input_ch = input_ch
        self._set_layers()
        self._set_activation()
        self.apply(self._init_weights)

    @abc.abstractmethod
    def forward(self):
        pass

    @abc.abstractmethod
    def _set_layers(self):
        pass

    @abc.abstractmethod
    def _init_weights(self):
        pass

    @abc.abstractmethod
    def _set_activation(self):
        pass

    @property
    def upscale_factor(self):
        return self._upscale_factor
