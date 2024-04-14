import numpy as np
from PIL import Image

def generate_image():
    # return 224x224 image + [angle_norm1, angle_norm2]
    # in which angle_norm in [-1, 1]
    # angle_norm = (rad - pi) / pi, rad in [0 2pi]

    # TODO: implement this function
    # for now just return a random image and two random angle_norm
    res_img, label = np.random.rand(224, 224), np.random.rand(2) * 2 - 1
    res_img = res_img.astype(np.uint8)
    res_img = Image.fromarray(res_img)

    return res_img, label