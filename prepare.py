
import cv2
import numpy as np

images = cv2.imread('hand.jpg')
image = cv2.imread('hand.jpg', 0)
# image = cv2.copyMakeBorder(
# image, 0, 0, 0, 0, cv2.BORDER_CONSTANT, None, value=0)

adjusted = cv2.convertScaleAbs(image, alpha=2.3, beta=11)
fil = cv2.bilateralFilter(adjusted, 5, 50, 55)
blur = cv2.medianBlur(fil, 13)
img_Th = cv2.adaptiveThreshold(
blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 47, 2)
blur2 = cv2.medianBlur(img_Th, 11)
kernel_img = np.ones((5, 5), np.uint8)
open_img = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, kernel_img)
New_image = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(New_image, cv2.COLOR_RGB2GRAY)
_, binary_img = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
contour, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imwrite("E:/BOA/prepare/original.jpg",images)
cv2.imwrite("E:/BOA/prepare/gray.jpg",image)
cv2.imwrite("E:/BOA/prepare/adjusted.jpg",adjusted)
cv2.imwrite("E:/BOA/prepare/fil.jpg",fil)
cv2.imwrite("E:/BOA/prepare/img_Th.jpg",img_Th)
cv2.imwrite("E:/BOA/prepare/blur2.jpg",blur2)
cv2.imwrite("E:/BOA/prepare/blur.jpg",blur)
cv2.imwrite("E:/BOA/prepare/open_img.jpg",open_img)
cv2.imwrite("E:/BOA/prepare/New_image.jpg",New_image)
cv2.imwrite("E:/BOA/prepare/binary_img.jpg",binary_img)

cv2.waitKey(0)
