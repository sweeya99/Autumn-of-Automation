import cv2
import numpy as np

img = cv2.imread('/home/panyam/Downloads/image153.jpg',0)
rows,cols = img.shape

M = np.float32([[1,0,50],[0,1,100]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()