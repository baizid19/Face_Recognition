import cv2
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('casecade_frontal.xml')
people = ['a rahim', 'abu sama', 'baizid', 'ikram', 'liton', 'mostazi', 'motiur', 'raihan', 'shajedur']

features = np.load('featuresByImage.npy', allow_pickle=True)
labels = np.load('labelsByImage.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trainedByImage.yml')

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(gray, 1.3, 3)
    for (x,y,w,h) in faces:
        faces_roi = gray[y:y + h, x:x + h] 

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')

        cv.rectangle(frame, (x - 22, y - 90), (x + w + 22, y - 22), (0, 155, 0), -1)
        cv.putText(frame, str(people[label]), (x,y-40), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv.imshow('Detect Face', frame)
    cv.waitKey(1)

cv2.release()
cv2.destroyAllWindows()