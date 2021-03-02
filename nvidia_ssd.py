from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.utils import source
from torch import Tensor
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from layers import Detect, PriorBox

Modules = List[nn.Module]


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', backbone_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=False)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=False)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=False)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class Nvidia_SSD(nn.Module):
    """
    Implementation of the Single Shot MultiBox Detector Architecture

    base:
        Standard ResNet
    extra:
        SSD layers conv8_1 through conv11_2, no layers between.
        Total layers: 8
    head:
        tuple of detection heads. (loc_head, conf_head)
    sizes:
        The dimentions and number of aspect ratios of the 6 feature layers
        layer 0:    38, 4
        layer 1:    19, 6
        layer 2:    10, 6
        layer 3:     5, 6
        layer 4:     3, 4
        layer 5:     1, 4
    """

    def __init__(self, phase: str, base: ResNet, cfg):
        super(Nvidia_SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg['num_classes']
        self.size = cfg['min_dim']
        self.default_boxes = cfg['default_boxes']
        self.priorbox = PriorBox(cfg)
        self.priors = self.priorbox.forward()

        self.base: ResNet = base
        self.extra = self._extra_layers(base.out_channels)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(self.num_classes, 0, 200, 0.01, 0.45)

        _loc_layers, _conf_layers = [], []
        for oc, nd in zip(self.base.out_channels, self.default_boxes):
            _loc_layers.append(nn.Conv2d(oc, 4 * nd, kernel_size=3, padding=1))
            _conf_layers.append(
                nn.Conv2d(oc, self.num_classes * nd, kernel_size=3, padding=1)
            )
        self.loc_head = nn.ModuleList(_loc_layers)
        self.conf_head = nn.ModuleList(_conf_layers)

    def _init_weights(self):
        layers = [*self.extra, *self.loc_head, *self.conf_head]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.
        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].
        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]
            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        x = self.base(x)

        source_layers = [x]
        for layer in self.extra:
            x = layer(x)
            source_layers.append(x)

        loc: List[Tensor] = []
        conf: List[Tensor] = []
        for x, l, c in zip(source_layers, self.loc_head, self.conf_head):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(
                    conf.view(conf.size(0), -1, self.num_classes)
                ),  # conf preds
                self.priors.type(type(x.data)),  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors,
            )
        return output

    def _extra_layers(self, in_channels: List[int]) -> nn.ModuleList:
        # Extra layers for feature scaling
        channels = [256, 256, 128, 128, 128]
        layers = []
        for i, (ins, outs, middles) in enumerate(
            zip(in_channels[:-1], in_channels[1:], channels)
        ):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(ins, middles, kernel_size=1, bias=False),
                    nn.BatchNorm2d(middles),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        middles, outs, kernel_size=3, padding=1, stride=2, bias=False
                    ),
                    nn.BatchNorm2d(outs),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(ins, middles, kernel_size=1, bias=False),
                    nn.BatchNorm2d(middles),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(middles, outs, kernel_size=3, bias=False),
                    nn.BatchNorm2d(outs),
                    nn.ReLU(inplace=True),
                )
            layers.append(layer)
        return nn.ModuleList(layers)

    def load_weights(self, base_file):
        print('Loading weights into state dict...')
        self.load_state_dict(
            torch.load(base_file, map_location=lambda storage, loc: storage)
        )
        print('Finished!')


def build_ssd(phase: str, cfg, basenet: str = '') -> Nvidia_SSD:
    base = ResNet(backbone='resnet34', backbone_path=basenet)
    return Nvidia_SSD(phase, base, cfg)
