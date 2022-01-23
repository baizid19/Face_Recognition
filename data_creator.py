import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

facedetect = cv2.CascadeClassifier('casecade_frontal.xml')

nameId = str(input("Enter Your Name: ")).lower()
path = 'dataset/'+nameId
isExist = os.path.exists(path)

if isExist:
    print("Name already taken")
    nameId = str(input("Please Enter Your Name Again: "))
else:
    os.makedirs((path))

count = 0
while True:
    _, frame = cap.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for (x,y,w,h) in faces:
        count = count+1
        name = './dataset/'+nameId+'/'+str(count)+'.jpg'
        cv2.imwrite(name, frame[y:y+h,x:x+w])
        print("Creating Image....."+name)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

    if count>150:
        break

cv2.release()
cv2.destroyAllWindows()



