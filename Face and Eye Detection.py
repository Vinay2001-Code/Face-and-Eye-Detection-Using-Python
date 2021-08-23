import numpy as np
import cv2

# Load the Cascade

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

 # CascadeClassifier is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.


cap = cv2.VideoCapture(0) # 0 is for Internal video camera, if we are using external we have to put 1

while 1:
    ret, img = cap.read()  # ret will obtain return value from getting the camera frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
   
    # Draw the rectangle around each face
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #A region of interest (ROI) is an area of an image defined for further analysis or processing
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)



    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()