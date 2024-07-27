import cv2
import numpy as np
import time
import poseEstimationModule as pm

path = "/home/user34/Desktop/gestureVolumeControl/videos/curl2.mp4"
cap = cv2.VideoCapture(path)

detector = pm.poseDetector()
count = 0
dir = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1080, 720))
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (200, 310), (0, 100))
        bar = np.interp(angle, (200, 310), (650, 100))
        # print(angle, per)
        # Check for the dumbbell curls
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        cv2.rectangle(img, (1080, 100), (1050, 650), (0, 255, 0), 3)
        cv2.rectangle(img, (1080, int(bar)), (1050, 650), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1080, 75), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        cv2.putText(img, f'{int(count)}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)
        
        

    cv2.imshow("Image", img)
    cv2.waitKey(15)


