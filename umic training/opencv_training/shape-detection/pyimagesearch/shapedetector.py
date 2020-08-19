# import the necessary packages
import cv2
#from __future__ import division

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = ".unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = ".triangle"
            
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			#(x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			print("1")
			print(ar)
			#shape = "square"if ar >= 0.95 and ar <= 1.05 else "rectangle"
			#if ar >= 0.95 and ar <= 1.05 :

			#	shape = "square"
		    #else :
			if  ar >= 0.95 and ar <= 1.05:
   				shape = "square"

			if ar <=0.95 :
				shape = "rhombus"
			if ar >= 1.05:
				shape = "rectangle"

            #leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            

		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"

		# otherwise, we assume the shape is a circle
		else:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			shape = ".circle" if ar >= 0.95 and ar <= 1.05 else ".oval"
		
		# return the name of the shape
		return shape
		'''
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)

			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
           		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = ".pentagon"

		# otherwise, we assume the shape is a circle
		
		else:
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			shape = ".circle" if ar >= 0.95 and ar <= 1.05 else ".oval"
			

		# return the name of the shape
		return shape
		'''