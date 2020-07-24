import cv2
import numpy as np

img = cv2.imread('/home/panyam/Downloads/Woman_7.jpg',0)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

surf = cv2.SURF(400)
kp, des = surf.detectAndCompute(img,None)
len(kp)

print (surf.hessianThreshold)
surf.hessianThreshold = 50000

kp, des = surf.detectAndCompute(img,None)
print( len(kp))
#sift = cv2.SIFT()
#kp = sift.detect(gray,None)

img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)
plt.imshow(img2),plt.show()

print (surf.upright)
surf.upright = True

kp = surf.detect(img,None)
img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4) 
plt.imshow(img2),plt.show()

print (surf.descriptorSize())

surf.extended
surf.extended = True

kp, des = surf.detectAndCompute(img,None)
print (surf.descriptorSize())
print (des.shape)

'''
img=cv2.drawKeypoints(gray,kp)

cv2.imwrite('sift_keypoints.jpg',img)

#img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/panyam/Downloads/Woman_7.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
'''