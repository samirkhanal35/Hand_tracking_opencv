import cv2
import numpy as np
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
cap.set(3, 1280)
cap.set(4, 720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

X1, Y1, X2, Y2 = 0, 0, 0, 0
prevx, prevy = 0, 0

point_reset_flag = False

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, channels = imgRGB.shape
    print("shape of image:", height, width)
    results = hands.process(imgRGB)
    print("-----------Results----------")
    thumb_tip = 4
    index_finger_tip = 8
    if results.multi_hand_landmarks:
        thumb_tip_landmark = {}
        index_finger_tip_landmark = {}
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

        # getting the tip values
        thumb_tip_landmark = handlms.landmark[thumb_tip]
        index_finger_tip_landmark = handlms.landmark[index_finger_tip]

        # X * width & Y * height
        X1 = int(thumb_tip_landmark.x * width)
        X2 = int(index_finger_tip_landmark.x * width)
        Y1 = int(thumb_tip_landmark.y * height)
        Y2 = int(index_finger_tip_landmark.y * height)

        # print("tip of thumb:", thumb_tip_landmark)
        # print("tip of index finger:", index_finger_tip_landmark)
        # print("X and y value of thumb tip", X1, Y1)
        # print("X and y value of index finger tip", X2, Y2)


        Distance = int(math.sqrt(((X2 - X1) ** 2) + ((Y2 - Y1) ** 2)))

        print("Distance between two fingers: ", Distance)

        if Distance<=30 :
            if point_reset_flag:
                point_reset_flag = False
                prevx, prevy = 0, 0
            else:
                point_reset_flag = True
            midx, midy = (X1+X2)//2, (Y1+Y2)//2
            cv2.circle(imgCanvas, (midx, midy), 10, (255, 0, 255), cv2.FILLED)
            if prevx == 0 and prevy == 0:
                prevx, prevy = midx, midy
            cv2.line(img, (prevx, prevy), (midx, midy), (0, 0, 255), 5)
            cv2.line(imgCanvas, (prevx, prevy), (midx, midy), (0, 0, 255), 5)
            prevx, prevy = midx, midy

    grayCanvas = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, invCanvas = cv2.threshold(grayCanvas, 48, 255, cv2.THRESH_BINARY_INV)
    invCanvas = cv2.cvtColor(invCanvas, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, invCanvas)
    img = cv2.bitwise_or(img, imgCanvas)

    # cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),2)


    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key & 0xFF == 27:
        cv2.imwrite("sample.jpg", img)
        break

cap.release()