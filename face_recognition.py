########################################################################
# File: face_recognition.py
# Author: @Lucasgb7 - lucasgb7_ks@hotmail.com
# Date: 07/21/2023
# Description: Face Recognition using Haar feature-based cascade classifiers
# : https://docs.opencv.org/4.8.0/db/d28/tutorial_cascade_classifier.html
#
# Collaboration: This example was based on the Mjrovai project
# : https://github.com/Mjrovai/OpenCV-Face-Recognition/
########################################################################
import cv2
import numpy as np
import os 

# Create a Local Binary Pattern reconizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Read the learned faces (dataset)
recognizer.read('trained_dataset.yml')
# Uses de Haar classifier to detect the faces
cascadePath = "haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
# Changing CV2 printint font
font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Lucas: id=1,  etc
names = ['None', 'Lucas', 'Luan'] 

# 480p video resolution: 640x480
cam = cv2.VideoCapture(0)
cam.set(3, 640) # width
cam.set(4, 480) # height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Detection inputs (scale, neighbors, size...)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    # Wait until 'ESC' is pressed to exit de video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\nByee!")
cam.release()
cv2.destroyAllWindows()