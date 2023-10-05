import torch

import os
import sys
sys.path.append(os.getcwd())


from eula.nn.centernet import CenterNet_MobilenetV3Small

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    model = CenterNet_MobilenetV3Small(num_classes=10)
    model._init_weights()
    x = torch.randn(1, 3, 64, 384)
    cls, offset = model(x)
    print(cls.shape, offset.shape)
    print(get_parameter_number(model))
