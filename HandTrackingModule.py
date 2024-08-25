import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionCon, self.trackCon)
    
    def findHands(self, img, highlight = set(), draw = True, draw_skeleton = True):
        highlight_ids = highlight
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        if draw and results.multi_hand_landmarks != None:
            for handLms in results.multi_hand_landmarks:
                if draw_skeleton: self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    if id in highlight_ids: cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

def main():
    pTime = 0
    cTime = 0

    detector = handDetector()
    cap = cv2.VideoCapture(0)
    highlight = set([4*i for i in range (0,6)])

    while True: 
        success, img = cap.read()
        detector.findHands(img, highlight)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()