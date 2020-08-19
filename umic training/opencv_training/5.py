'''import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_eye.xml')
#face_cascade.load('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

#test = face_cascade.load('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

#print(test)


img = cv2.imread('/home/panyam/Downloads/sweeya photo.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow('gray',gray)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#print(faces)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    print('2')
    cv2.imshow('img',img)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/panyam/opencv_workspace/opencv/data/haarcascades/haarcascade_eye.xml')

img = cv2.imread('/home/panyam/Downloads/Woman_7.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#faces = face_cascade.detectMultiScale(gray)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
