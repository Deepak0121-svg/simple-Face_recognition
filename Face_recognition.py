import os
import cv2
import face_recognition
from datetime import datetime
import pandas as pd
import numpy as np

path = 'path'#your path here
images = []
classNames = []

# Get a list of directories in the dataset folder
myList = os.listdir(path)

for cl in myList:
    # Construct the full path and check if it's a directory
    curDir = os.path.join(path, cl)
    if os.path.isdir(curDir):
        curImgList = os.listdir(curDir)
        for img in curImgList:
            imgPath = os.path.join(curDir, img)
            currentImg = cv2.imread(imgPath)
            if currentImg is None:
                print(f"Warning: {imgPath} could not be loaded.")
                continue
            images.append(currentImg)
            classNames.append(cl)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Ensure that the list is not empty
            encodeList.append(encodings[0])
        else:
            print("Warning: No face detected in one of the images.")
    return encodeList

encodeListKnown = findEncodings(images)
print('Encodings Complete')

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f: #your csv file here

        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read(0)
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




