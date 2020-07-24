import cv2
import numpy as np

img_rgb = cv2.imread("/home/panyam/Downloads/sweeya photo.png")
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

img_gray_inv = 255 - img_gray
img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                            sigmaX=0, sigmaY=0)


def dodgeNaive(image, mask):
  # determine the shape of the input image
  width,height = image.shape [:2] 

  # prepare output argument with same size as image
  blend = np.zeros((width,height), np.uint8)

  for col in xrange(width):
    for row in xrange(height):
      # do for every pixel  
      if mask[c,r] == 255:
        # avoid division by zero 
        blend[c,r] = 255
      else:
        # shift image pixel value by 8 bits
        # divide by the inverse of the mask
        tmp = (image[c,r] << 8) / (255-mask)

        # make sure resulting value stays within bounds
        if tmp > 255:
          tmp = 255
          blend[c,r] = tmp

  return blend                            

def dodgeV2(image, mask):
        return cv2.divide(image, 255-mask, scale=256)

def burnV2(image, mask):
        return 255 - cv2.divide(255-image, 255-mask, scale=256)


img_blend = dodgeV2(img_gray, img_blur)
cv2.imshow("pencil sketch", img_blend)

cv2.waitKey(0)
cv2.destroyAllWindows()
img_canvas = cv2.imread("/home/panyam/Downloads/sweeya photo.png")
img_blend = cv2.multiply(img_blend, img_canvas, scale=1/256)

cv2.waitKey(0)
cv2.destroyAllWindows()