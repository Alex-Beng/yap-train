import math

import torch
import torch.nn as nn

from .backbone import MobileNetV3Small
from .head import MobileNetV3Small_head


class CenterNet_MobilenetV3Small(nn.Module):
    def __init__(self, num_classes = 2) -> None:
        super().__init__()
        self.backbone = MobileNetV3Small(64)
        self.head = MobileNetV3Small_head(num_classes, 64)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * \
                    m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
        self.head.cls_head[-1].bias.data.fill_(0)
        self.head.reg_head[-1].bias.data.fill_(-2.19)
    def forward(self, x):
        x = self.backbone(x)
        hm, offset = self.head(x)
        return hm, offset
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def detect_image(self, image, device):
        image = image.to(device)
        image = image.unsqueeze(0)
        
        with torch.no_grad():
            hm, offset = self.forward(image)
            
        
        return hm, offset