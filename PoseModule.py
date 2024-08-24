import cv2
import mediapipe as mp
import time
import math


class poseDetector():

    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, 
                 smooth_segmentation=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, 
                                     self.smooth_segmentation, self.min_detection_confidence, self.min_tracking_confidence)

    def findPose(self, img, highlighted = set(), draw=True):
        highlighted_id = highlighted
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        self.lmList = []
        if results.pose_landmarks:
            if draw: self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if id in highlighted_id: cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def checkHandsJoined(self, img, draw=True):
        if len(self.lmList) == 0: return None
        right_wrist = self.lmList[self.mpPose.PoseLandmark.RIGHT_WRIST][1:]
        left_wrist = self.lmList[self.mpPose.PoseLandmark.LEFT_WRIST][1:]
        euclidean_distance = int(math.hypot(left_wrist[0]-right_wrist[0], left_wrist[1]-right_wrist[1]))
        self.handStauts = None
        if euclidean_distance < 130:
            self.hand_status = "Hands Joined"
            color = (0, 255, 0)
        else:
            self.hand_status = "Hands Not Joined"
            color = (0, 0, 255)
        
        if draw:
            cv2.putText(img, self.hand_status, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
        
        return self.hand_status

    def checkLeftRight(self, img, draw=True):
        if len(self.lmList) == 0: return None
        h = img.shape[0]
        mid = img.shape[1]//2
        self.horizontal_position = None
        left_shoulder = self.lmList[self.mpPose.PoseLandmark.RIGHT_SHOULDER]
        right_shoulder = self.lmList[self.mpPose.PoseLandmark.LEFT_SHOULDER]
        mid_shoulder = (left_shoulder[1]+right_shoulder[1])//2
        left_x = (left_shoulder[1]+mid_shoulder)//2
        right_x = (right_shoulder[1]+mid_shoulder)//2

        if (right_x <= mid and left_x <= mid):
            self.horizontal_position = 'Left'
        elif (right_x >= mid and left_x >= mid):
            self.horizontal_position = 'Right'
        elif (right_x >= mid and left_x <= mid):
            self.horizontal_position = 'Center'
        
        if draw:
            cv2.putText(img, self.horizontal_position, (5, h-10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            cv2.line(img, (mid, 0), (mid, h), (255, 255, 255), 2)
            cv2.circle(img, (left_x, left_shoulder[2]), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (right_x, right_shoulder[2]), 5, (0, 255, 0), cv2.FILLED)
            
        return self.horizontal_position

    def checkJumpCrouch(self, img, threshold=250, draw=True):
        if len(self.lmList) == 0: return None
        h, w, c = img.shape
        left_y = int(self.lmList[self.mpPose.PoseLandmark.RIGHT_SHOULDER][2])
        right_y = int(self.lmList[self.mpPose.PoseLandmark.LEFT_SHOULDER][2])
        mid_y = abs(right_y + left_y) // 2
        lower_bound = threshold+45
        upper_bound = threshold-45
        self.posture = None
        if mid_y < upper_bound: self.posture = 'Jumping'
        elif mid_y > lower_bound: self.posture = 'Crouching'
        else: self.posture = 'Standing'
        if draw:
            cv2.putText(img, self.posture, (5, h-50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
            cv2.line(img, (0, lower_bound), (w, lower_bound), (255, 255, 255), 2)
            cv2.line(img, (0, upper_bound), (w, upper_bound), (255, 255, 255), 2)
        return self.posture

def main():
    detector = poseDetector()
    cap = cv2.VideoCapture(0)
    pTime = 0
    highlighted = set([5*i for i in range(5)])
    while True:
        success, img = cap.read()
        if not success: continue
        detector.findPose(img, highlighted)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
