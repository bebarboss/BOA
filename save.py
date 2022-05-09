
import cv2
import numpy as np
import os
import uuid
from crop import result



IMAGE_PATH='E:/BOA/data/image'
labels=['gongcogang']

img_counter = 0
x=0

for label in labels:
    os.mkdir ('E:/BOA/data/image \\'+label)
while True:
    if x<360:
        x=x+3.6
        (h, w) = result.shape[:2]
        center = (w / 2, h / 2)
        scale = 1

        print('collecting {}'.format(x))
        for imgnum in range(1):

            M = cv2.getRotationMatrix2D(center, x, scale)
            rotated = cv2.warpAffine(result, M, (w, h))
            imagename=os.path.join(IMAGE_PATH,label,label+'{}.jpg'.format(str(uuid.uuid1())))  
  

            cv2.imwrite(imagename,rotated) 
            img_counter += 1

            cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    

