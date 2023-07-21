########################################################################
# File: face_training.py
# Author: @Lucasgb7 - lucasgb7_ks@hotmail.com
# Date: 07/21/2023
# Description: Train the gotten dataset into .yml file
#
# Collaboration: This example was based on the Mjrovai project
# : https://github.com/Mjrovai/OpenCV-Face-Recognition/
########################################################################
import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset/images'

# Create a Local Binary Pattern reconizer 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    # From the path, it separates the users by the id
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:
        # Converts image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        # Get the user id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        # Append the face to the user id
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\nLearning the new faces, wait a moment...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into .\
recognizer.write('trained_dataset.yml')

# Print the numer of faces trained and end program
print("\nFaces learned and stored!".format(len(np.unique(ids))))