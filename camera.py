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
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    M = cv2.moments(binary)

    if M['m00'] != 1:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            
            mask = np.zeros(New_image.shape, dtype=np.uint8)
            cv2.circle(mask, (cx,cy), 170, (255,255,255), -1)
            ROI = cv2.bitwise_and(New_image, mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            x,y,w,h = cv2.boundingRect(mask)
            result = ROI[y:y+h,x:x+w]
            mask = mask[y:y+h,x:x+w]
            result[mask==0] = (0,0,0) 
 
    
    if not ret:
        print("failed to grab frame")
        break
    
    cv2.imshow('original',frame)
    

    k = cv2.waitKey(1)
    
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        for label in labels:
            imagename=os.path.join(IMAGE_PATH,label,label+'{}.jpg'.format(str(uuid.uuid1())))
            img=cv2.imwrite(imagename, frame)
            print("{} written!".format(imagename))

cap.release()