import cv2

img_path = "./assets/FFF.bmp"
img = cv2.imread(img_path)

print(img.shape)

# convert to L channel
img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
# split channels
l, a, b = cv2.split(img)

# threshold in L
_, l = cv2.threshold(l, 100, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)

cv2.imshow("l", l)
cv2.waitKey()

# find the largest contour 
# and fill it with white
contours, _ = cv2.findContours(l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(l, contours, -1, 255, cv2.FILLED)

cv2.imshow("mask", l)
cv2.waitKey()

# save the mask
cv2.imwrite("mask.png", l)

# get mask with only F area(biggest) and its little triangle(second biggest)
contours, _ = cv2.findContours(l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
F_area_contours = contours[-1]
triangle_contours = contours[-2]

F_area_contours_bbox = cv2.boundingRect(F_area_contours)
triangle_contours_bbox = cv2.boundingRect(triangle_contours)

import numpy as np

F_area_mask = cv2.drawContours(np.zeros_like(l), [F_area_contours], -1, 255, cv2.FILLED)
triangle_mask = cv2.drawContours(np.zeros_like(l), [triangle_contours], -1, 255, cv2.FILLED)

new_image = cv2.imread(img_path)

# 通过mask和bbox获取F区域的图像
F_area = cv2.bitwise_and(new_image, new_image, mask=F_area_mask)
F_area = F_area[F_area_contours_bbox[1]:F_area_contours_bbox[1]+F_area_contours_bbox[3], F_area_contours_bbox[0]:F_area_contours_bbox[0]+F_area_contours_bbox[2]]


triangle_area = cv2.bitwise_and(new_image, new_image, mask=triangle_mask)
triangle_area = triangle_area[triangle_contours_bbox[1]:triangle_contours_bbox[1]+triangle_contours_bbox[3], triangle_contours_bbox[0]:triangle_contours_bbox[0]+triangle_contours_bbox[2]]

print(F_area.shape, triangle_area.shape)


# read a large image
b_img_path = "./assets/test.png"
b_img = cv2.imread(b_img_path)

r, c, _ = b_img.shape
res_w, res_h = 67, 380

for _ in range(10):
        
    # random crop the b_img
    x = np.random.randint(0, c-res_w)
    y = np.random.randint(0, r-res_h)
    # need deepcopy
    res_img = b_img[y:y+res_h, x:x+res_w].copy()
    

    cv2.imshow("res_img", res_img)
    cv2.waitKey()

    '''
    +------+
    |      |
    |  F > | # > shift in datagen
    |      |
    |      |
    |      |
    |      |
    +------+
    '''
    F_x = np.random.randint(0, 3)
    F_y = np.random.randint(0, res_h-33)

    tri_x = np.random.randint(F_x + 40 + 1, res_w-10)
    tri_y = np.random.randint(F_y + 1, F_y + 33 - 17 -1)

    # paste F_area and triangle_area to res_img
    res_img[F_y:F_y+33, F_x:F_x+40] = F_area
    res_img[tri_y:tri_y+17, tri_x:tri_x+10] = triangle_area

    cv2.imshow("res_img", res_img)
    cv2.waitKey()

