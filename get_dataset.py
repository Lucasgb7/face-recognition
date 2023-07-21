########################################################################
# File: get_dataset.py
# Author: @Lucasgb7 - lucasgb7_ks@hotmail.com
# Date: 07/21/2023
# Description: Take pictures using the webcam to gathering on a
# dataset file. Uses the face detection algorithm.
# 
# Collaboration: This example was based on the Mjrovai project
# : https://github.com/Mjrovai/OpenCV-Face-Recognition/
########################################################################
import cv2
import os

cam = cv2.VideoCapture(0)
# 480p video resolution: 640x480
cam.set(3, 640) # width
cam.set(4, 480) # height
# Use Haarcascade Classifier Training
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
print("\n----------------------------------")
face_id = input("\n Give the USER ID (>0): ")
print("\nInitializing camera... Look and wait...")
print("--> Press 'ESC' to exit.")
# Initialize individual sampling count and a sample limit
count = 0
samples_number = 30

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    # On the 
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/images/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)

    # Wait until 'ESC' is pressed to exit de video
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= samples_number:
         break

# Do a bit of cleanup
print("\nExiting program!!")
print("\n----------------------------------")
cam.release()
cv2.destroyAllWindows()

