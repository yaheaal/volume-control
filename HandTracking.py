import mediapipe as mp
import cv2

class handDetector():
    def __init__(self,
                mode=False,
                max_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):
    
        self.mode = mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.max_hands, 0,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame, draw=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        lms = self.results.multi_hand_landmarks

        if lms and draw:
            for handlms in lms:
                self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self, frame):
        lmList = []
        lms = self.results.multi_hand_landmarks

        if lms:
            thisHand = lms[0]
            for id, lm in enumerate(thisHand.landmark):
                h, w, c = frame.shape
                cX, cY = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cX, cY])
        return lmList
