import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers, cfgs

def make_layers(cfg, in_channels=3, batch_norm=False, dilation=1):
    
    layers = []
    if dilation:
        d_rate = 2

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=dilation, dilation=dilation)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class CSRNet(nn.Module):
    def __init__(self, load_weight=False):
        super(CSRNet, self).__init__()
        self.frontend_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.backend_cfg = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_cfg, batch_norm=True)
        self.backend = make_layers(self, in_channels=512, batch_norm=True, dilation=2)
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)