import cv2
import matplotlib.pyplot as plt
import numpy as np

nemo = cv2.imread('/home/panyam/Downloads/image153.jpg')
plt.imshow(nemo)
plt.show()
'''
originalImage = cv2.imread('/home/panyam/Downloads/sweeya photo.png')
red2blue_imag = cv2.cvtColor(originalImage, flag)


cv2.imshow('red2blue_imag', red2blue_imag)
cv2.imshow('Original image',originalImage)
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print (flags)


cv2.waitKey(0)
cv2.destroyAllWindows()

'''