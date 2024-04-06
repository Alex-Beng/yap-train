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

# find the largest contour and its bbox
contours, _ = cv2.findContours(l, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
print(len(contours))
contours_bbox = cv2.boundingRect(contours[-2])

# crop image with bbox and save
new_image = cv2.imread(img_path)
new_image = new_image[contours_bbox[1]:contours_bbox[1]+contours_bbox[3], contours_bbox[0]:contours_bbox[0]+contours_bbox[2]]
cv2.imwrite("crop.png", new_image)