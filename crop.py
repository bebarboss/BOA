import cv2
import numpy as np
from prepare import New_image,contour

for i in contour:
        M = cv2.moments(i)
        if M['m00'] != 0:
                cx = int(M['m10']/M['m00']) 
                cy = int(M['m01']/M['m00'])
                
                mask = np.zeros(New_image.shape, dtype=np.uint8)
                cv2.circle(mask, (cx,cy+50), 130, (255,255,255), -1)
                ROI = cv2.bitwise_and(New_image, mask)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                x,y,w,h = cv2.boundingRect(mask)
                result = ROI[y:y+h,x:x+w]
                mask = mask[y:y+h,x:x+w]
                result[mask==22] = (0,0,0) 
print(f"x: {cx} y: {cy}")

cv2.imshow("crop",result)
cv2.imwrite("E:/BOA/prepare/crop.jpg",result)

cv2.waitKey(0)




