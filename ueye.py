import cv2
import numpy as np
import os
import uuid



cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,1024)
cap.set(3, 4)

IMAGE_PATH='E:/BOA/data'
labels=[""]

img_counter = 0


while True:
    ret, frame = cap.read()
    roi=frame[:1080,0:1920]
    resize = cv2.resize(frame, (1080,1920))
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    adjusted = cv2.convertScaleAbs(gray, alpha=1, beta=11)
    fil = cv2.bilateralFilter(adjusted, 5, 50, 55)
    blur = cv2.medianBlur(fil, 11)
    img_Th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 33, 1)
    blur2 = cv2.medianBlur(img_Th, 13)
    kernel_img = np.ones((5, 5), np.uint8)
    open_img = cv2.morphologyEx(blur2, cv2.MORPH_OPEN, kernel_img)
    New_image = cv2.cvtColor(open_img, cv2.COLOR_BGR2RGB)
    cv2.imshow("s",New_image)

    