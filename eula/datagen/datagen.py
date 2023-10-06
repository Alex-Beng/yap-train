import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from .img_bytes import F_area_bytes, tri_area_bytes
import cv2
from PIL import Image
import numpy as np
import random

F_area_array = np.frombuffer(F_area_bytes, dtype=np.uint8)
F_area = cv2.imdecode(F_area_array, cv2.IMREAD_COLOR)

tri_area_array = np.frombuffer(tri_area_bytes, dtype=np.uint8)
tri_area = cv2.imdecode(tri_area_array, cv2.IMREAD_COLOR)

background_images = []



# centernet 所需的高斯分布半径
def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

# 绘制heat map需要的高斯分布
def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

class EulaDataset(Dataset):
    def __init__(self, num_classes, length, input_shape, bg_images) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.length = length

        self.input_shape = input_shape
        self.output_shape = (input_shape[0] // 4, input_shape[1] // 4)
        print(f'output_shape: {self.output_shape}, input_shape: {self.input_shape}')
        self.bg_images = bg_images

    def __len__(self):
        return self.length
    
    def generate_image(self):
        # raw: 380x67 image and (F_label_x, F_label_y)
        # return: 384x64 and its label

        rand_bg = random.choice(self.bg_images)
        
        bg_r, bg_c, _ = rand_bg.shape
        res_w, res_h = 67, 380

        # random crop the b_img
        x = np.random.randint(0, bg_c-res_w)
        y = np.random.randint(0, bg_r-res_h)
        # need deepcopy
        res_img = rand_bg[y:y+res_h, x:x+res_w].copy()

        F_x = np.random.randint(0, 3)
        F_y = np.random.randint(0, res_h-33)

        tri_x = np.random.randint(F_x + 40 + 1, res_w-10)
        tri_y = np.random.randint(F_y + 1, F_y + 33 - 17 -1)

        # paste F_area and triangle_area to res_img
        res_img[F_y:F_y+33, F_x:F_x+40] = F_area
        res_img[tri_y:tri_y+17, tri_x:tri_x+10] = tri_area

        F_label_x = F_x + 40//2
        F_label_y = F_y + 33//2

        # convert to 384x64
        res_img = cv2.resize(res_img, (64, 384), interpolation=cv2.INTER_AREA)

        F_label_x = F_label_x * 64 // 67
        F_label_y = F_label_y * 384 // 380

        # convert to RGB
        res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
        # print(f'res img shape {res_img.s}')
        res_img = Image.fromarray(res_img)
        
        return res_img, (F_label_x, F_label_y)


    def __getitem__(self, index):
        img, label = self.generate_image()

        # 生成 centernet 结果
        hm  = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)
        print(hm.shape)
        # here only Fkey&tri is 1, bg is 0
        cls_id = 1

        h, w = 20, 20 # hard code it
        radius = gaussian_radius((h, w), min_overlap=0.7)
        radius = max(0, int(radius))

        center = np.array(label, dtype=np.float32)
        center[0] = center[0] / self.input_shape[1] * self.output_shape[1]
        center[1] = center[1] / self.input_shape[0] * self.output_shape[0]
        center_int = center.astype(np.int32)
        
        print(center, radius, label, center_int)
        hm[:, :, cls_id] = draw_gaussian(hm[:, :, cls_id], center_int, radius)
        reg[center_int[1], center_int[0]] = center - center_int
        reg_mask[center_int[1], center_int[0]] = 1
        
        # convert img to [0, 1]
        img = transforms.ToTensor()(img)

        return img, hm

    
