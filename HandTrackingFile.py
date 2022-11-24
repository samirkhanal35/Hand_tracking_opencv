import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


        print("tip of thumb:", thumb_tip_landmark)
        print("tip of index finger:", index_finger_tip_landmark)



    # cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,255),2)


    cv2.imshow("img", img)
    cv2.waitKey(1)
