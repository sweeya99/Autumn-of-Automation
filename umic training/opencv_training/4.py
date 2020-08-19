import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX
orig = cv2.imread("/home/panyam/Desktop/4.png")


img = cv2.imread("/home/panyam/Desktop/4.png",  cv2.IMREAD_GRAYSCALE )
_, threshold = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

print(contours)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    cv2.drawContours(orig, [approx], 0, (0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1]
    #M = cv2.moments(cnt)
	#x = int((M["m10"] / M["m00"]) * ratio)
	#y = int((M["m01"] / M["m00"]) * ratio)
    if len(approx) == 3:
        cv2.putText(orig, "Triangle", (x, y), font, 1, (0))

    elif len(approx) == 4:
        cv2.putText(orig, "Rectangle", (x, y), font, 1, (0))
    elif len(approx) == 5:
        cv2.putText(orig, "Pentagon", (x, y), font, 1, (0))
    elif 6 < len(approx) < 15:
        cv2.putText(orig, "Oval", (x, y), font, 1, (0))
    else:
        cv2.putText(orig, "Circle", (x, y), font, 1, (0))



cv2.imshow("orig",orig)
#cv2.imshow("shapes", img)
#cv2.imshow("Threshold", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()        

'''
# USAGE
# python detect_shapes.py --image shapes_and_colors.png

# import the necessary packages
from pyimagesearch.shapedetector import ShapeDetector
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
image = cv2.imread(args["image"])
resized = imutils.resize(image, width=300)
ratio = image.shape[0] / float(resized.shape[0])

# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
sd = ShapeDetector()

#print(cnts)
# loop over the contours
for c in cnts:
	print(c)
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
	M = cv2.moments(c)
	cX = int((M["m10"] / M["m00"]) * ratio)
	cY = int((M["m01"] / M["m00"]) * ratio)
	shape = sd.detect(c)
 	print('1')

	# multiply the contour (x, y)-coordinates by the resize ratio,
	# then draw the contours and the name of the shape on the image
	c = c.astype("float")
	c *= ratio
	c = c.astype("int")
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
	cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
		0.5, (255, 255, 255), 2)
		

	# show the output image
cv2.imshow("Image", image)
	#cv2.imshow("blurred", blurred)
	#cv2.imshow("thresh ", thresh )
cv2.waitKey(0)
cv2.destroyAllWindows()
'''