"""SRCNN architecture implementation

Reference is here:
https://arxiv.org/abs/1501.00092

"""

from torch import nn

from .model import MetaNet


class SRCNN(MetaNet):

    def __init__(self, upscale_factor, input_ch=1, second_layer_kernel=(5, 5)):
        self._second_layer_kernel = second_layer_kernel
        self._input_ch = input_ch
        super().__init__(upscale_factor, input_ch)

    def _set_layers(self):
        self.conv1 = nn.Conv2d(self._input_ch, 64, kernel_size=(9, 9), stride=1, padding=0, padding_mode='zeros')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=self._second_layer_kernel, stride=1, padding=0, padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, self._input_ch, kernel_size=(3, 3), padding=0, padding_mode='zeros')

    def _set_activation(self):
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.constant_(layer.bias, val=0)
