"""MCH architecture implementation


Reference is here:
https://doi.org/10.1587/transfun.E100.A.572
http://www.lib.kobe-u.ac.jp/handle_kernel/90006266

In above references, the initialization algorithm of layers are originally not xavier
"""

from typing import Dict, List, Tuple

from torch import nn

from .model import MetaNet

SCALE_KERNELSIZES = {
    2: [(5, 5), (5, 5), (3, 3)],
    3: [(5, 5), (3, 3), (1, 1)],
    4: [(5, 5), (3, 3), (1, 1)],
}


class MCH(MetaNet):

    def __init__(self, upscale_factor: int, input_ch: int,
                 scale_kernelsizes: Dict[int, List[Tuple[int, int]]] = SCALE_KERNELSIZES):
        self._kernel_sizes = scale_kernelsizes[upscale_factor]
        super().__init__(upscale_factor, input_ch)

    def _set_layers(self):
        self.conv1 = nn.Conv2d(
            self._input_ch, 64, kernel_size=self._kernel_sizes[0], stride=1, padding=2, padding_mode='zeros')
        self.conv2 = nn.Conv2d(64, 32, kernel_size=self._kernel_sizes[1], stride=1,
                               padding=2, padding_mode='zeros')
        self.conv3 = nn.Conv2d(32, self._upscale_factor ** 2,
                               kernel_size=self._kernel_sizes[2], padding=1, padding_mode='zeros')
        self.ps = nn.PixelShuffle(self._upscale_factor)

    def _set_activation(self):
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.ps(x)
        return x

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_normal_(layer.weight)
            nn.init.constant_(layer.bias, val=0)
