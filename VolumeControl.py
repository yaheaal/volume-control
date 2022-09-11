import cv2
import time
import math
import numpy as np
import osascript
import HandTracking as hd

minVol = 0
maxVol = 100

minDis = 50
maxDis = 300

vol = 0
volBar = 600
volPer = 0

detector = hd.handDetector(min_detection_confidence=.8)

cap = cv2.VideoCapture(0)

while True:
    t0 = time.time()
    _, frame = cap.read()

    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)

    if len(lmList):
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cX, cY = (x1+x2)//2, (y1+y2)//2

        cv2.circle(frame, (x1, y1), 15, (0,255,0), -1)
        cv2.circle(frame, (x2, y2), 15, (0,255,0), -1)
        cv2.circle(frame, (cX, cY), 15, (0,255,0), -1)
        cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 3)
        
        length = math.hypot(x2-x1, y2-y1)
        vol = np.interp(length, [minDis,maxDis], [minVol,maxVol])
        volBar = np.interp(length, [minDis,maxDis], [600,150])
        volPer = np.interp(length, [minDis,maxDis], [0,100])
        osascript.osascript("set volume output volume {}".format(vol))

        if length<minDis:
            cv2.circle(frame, (cX, cY), 15, (0,0,255), -1)
    
    cv2.rectangle(frame, (minDis, 150), (85, 600), (0,255,0), 3)
    cv2.rectangle(frame, (minDis, int(volBar)), (85, 600), (0,255,0), -1)
    cv2.putText(frame, f"{int(volPer)} %", (45,650), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    t1 = time.time()
    fps = 1/(t1-t0)
    cv2.putText(frame, f"FPS: {int(fps)}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Cam', frame)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

