import mediapipe as mp
import HandTracking as ht
import cv2
import time

def main():
    cap = cv2.VideoCapture(0)
    detector = ht.handDetector()
    
    while True:
        t0 = time.time()
        _, frame = cap.read()
        frame = detector.findHands(frame)
        # lmList = detector.findPosition(frame)

        t1 = time.time()
        fps = 1/(t1-t0)
        cv2.putText(frame, f"FPS: {int(fps)}", (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)

        cv2.imshow('Cam', frame)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()