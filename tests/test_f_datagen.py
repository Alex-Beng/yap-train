import os
import sys
sys.path.append(os.getcwd())

from eula.datagen.datagen import EulaDataset

import cv2
import numpy as np

if __name__ == "__main__":
    bg_imgs = [cv2.imread("./assets/test.png")]


    dataset = EulaDataset(3, 10, (384, 64), bg_imgs)
    for i in range(10):
        img, hm = dataset[0]
        
        img *= 255
        img = np.array(img)
        # print(img.shape)
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        cv2.imshow("img", img)
        # cv2.waitKey()
        print(np.max(hm), np.min(hm), hm.shape)
        hm *= 255
        hm = np.array(hm)
        # hm = np.transpose(hm, (1, 2, 0))
        img = img.astype(np.uint8)
        cv2.imshow("hm", hm)
        cv2.waitKey()
    exit()
    hm0 = hm[:, :, 0]
    hm1 = hm[:, :, 1]

    # 扔掉最后一维
    hm0 = hm0.astype(np.uint8)
    hm1 = hm1.astype(np.uint8)
    hm0 = np.transpose(hm0, (1, 0))
    hm1 = np.transpose(hm1, (1, 0))
    print(hm0.shape)
    cv2.imshow("hm0", hm0)
    cv2.imshow("hm1", hm1)
    cv2.waitKey()