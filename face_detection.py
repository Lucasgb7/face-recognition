########################################################################
# File: face_detection.py
# Author: @Lucasgb7 - lucasgb7_ks@hotmail.com
# Date: 07/21/2023
# Description: Face Detection using Haar feature-based cascade classifiers
# : https://docs.opencv.org/4.8.0/db/d28/tutorial_cascade_classifier.html
#
# Collaboration: This example was based on the Mjrovai project
# : https://github.com/Mjrovai/OpenCV-Face-Recognition/
########################################################################
import numpy as np
import cv2

# Loads the classifier from OpenCV files (https://github.com/opencv/opencv/tree/4.x/data/haarcascades)
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
# Set resolution to 480p (640x480)
cap.set(3,640) # width
cap.set(4,480) # height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
    #   gray is the input grayscale image.
    #   scaleFactor is the parameter specifying how much the image size is reduced at each image scale. It is used to create the scale pyramid.
    #   minNeighbors is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
    #   minSize is the minimum rectangle size to be considered a face.

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    
    # OpenCV shows camera on real-time
    cv2.imshow('video',img)

    # Wait until 'ESC' is pressed to exit de video
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()