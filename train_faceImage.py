import numpy as np
import cv2 as cv
import os

people = ['a rahim', 'abu sama', 'baizid', 'ikram', 'liton', 'mostazi', 'motiur', 'raihan', 'shajedur']

DIR = r'C:\Users\mdbai\PycharmProjects\pythonProject\dataset'
haar_cascade = cv.CascadeClassifier('casecade_frontal.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("train done.......")
features = np.array(features, dtype = 'object')
face_recognizer = cv.face.LBPHFaceRecognizer_create()

#train the recognizer on the feature list and label list
face_recognizer.train(features, np.array(labels))

face_recognizer.save('face_trainedByImage.yml')
np.save('featuresByImage.npy', features)
np.save('labelsByImage.npy', labels)