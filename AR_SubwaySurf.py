import cv2
import numpy as np
import time
import PoseModule as pm
import pyautogui as pg

cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cv2.namedWindow('AR_SubwaySurf', cv2.WINDOW_NORMAL)
detector = pm.poseDetector()
game_started = False
x_pos_index = 1
y_pos_index = 1
threshold = None
counter = 0
num_frames = 20

while cap.isOpened():
    success, img = cap.read()
    if not success: continue
    img = cv2.flip(img, 1)
    h, w, c= img.shape
    detector.findPose(img, draw = game_started)
    if game_started:
        horizontal_pos = detector.checkLeftRight(img)
        vertical_pos = detector.checkJumpCrouch(img, threshold)
        if (horizontal_pos=='Left' and x_pos_index!=0) or (horizontal_pos=='Center' and x_pos_index==2):
            pg.press('left')
            x_pos_index -= 1
        elif (horizontal_pos=='Right' and x_pos_index!=2) or (horizontal_pos=='Center' and x_pos_index==0):
            pg.press('right')
            x_pos_index += 1
        if (vertical_pos=='Jumping' and y_pos_index==1):
            pg.press('up')
            y_pos_index += 1
        elif (vertical_pos=='Crouching' and y_pos_index==1):
            pg.press('down')
            y_pos_index -= 1
        elif (vertical_pos=='Standing' and y_pos_index!=1):
            y_pos_index = 1
        if detector.checkHandsJoined(img) == 'Hands Joined':
            pg.press('space')
            time.sleep(1)
    else:
        cv2.putText(img, 'JOIN BOTH HANDS TO START THE GAME', (5, h - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        if detector.checkHandsJoined(img) == 'Hands Joined':
            counter += 1
            if counter == num_frames:
                game_started = True
                left_y = int(detector.lmList[detector.mpPose.PoseLandmark.RIGHT_SHOULDER][2])
                right_y = int(detector.lmList[detector.mpPose.PoseLandmark.LEFT_SHOULDER][2])
                threshold = abs(right_y + left_y) // 2
        else:
            counter = 0
    
    cv2.imshow('AR_SubwaySurf', img)
    k = cv2.waitKey(1)
    if k == 27: break

cap.release()
cv2.destroyAllWindows()