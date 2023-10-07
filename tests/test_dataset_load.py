
import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2



ds_path = "./data/eula/train.pt"
ds = torch.load(ds_path)

dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

for dt in dl:
    img, hm, reg, reg_msk = dt
    img *= 255
    img = np.array(img).reshape(3, 384, 64)
    print(img.shape)
    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.uint8)
    hm *= 255
    print(hm.shape)
    hm = np.array(hm).reshape(96, 16, 2)[:, :, 1]
    print(hm.shape)
    cv2.imshow("hm", hm)
    cv2.imshow("img", img)
    cv2.waitKey()