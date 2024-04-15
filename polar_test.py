import cv2
import numpy as np
from random import randint

# r in [70, 110]
# v_angle in [70, 90]
for r in range(0, 130, 10):
    flipped = np.ones(shape=(360, 20), dtype=np.uint8)

    cv2.imshow('flipped',flipped)

    h,w = flipped.shape

    radius = r
    print(radius)

    new_image = np.zeros(shape=(h,radius+w),dtype=np.uint8)
    h2,w2 = new_image.shape

    v_angle = randint(70, 90)
    v_beg = randint(0, 360 - v_angle)
    
    # set the new_image
    new_image[v_beg:v_beg+v_angle, :] = 255


    new_image[: ,w2-w:w2] = flipped
    # new_image = ~new_image
    print(new_image.shape)

    cv2.imshow('polar',new_image)

    h,w = new_image.shape

    center = (112, 112) 

    maxRadius = 112

    output= cv2.warpPolar(new_image, center=center, maxRadius=radius, dsize=(maxRadius*2,maxRadius*2), flags=cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
    print(output.shape)

    cv2.imshow('output',output)
    cv2.waitKey(0)
