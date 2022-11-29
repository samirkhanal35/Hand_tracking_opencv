import cv2
import mediapipe as mp
import time


class simpleHandDetector():

    def __init__(self, mode=False):
        self.mode = mode
        # self.maxhands = maxhands
        # self.detectCon = detectCon
        # self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(mode)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handlms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def fingerPos(self, frame, handNo, draw=False):
        self.landmark_list = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):

                h, w, c = frame.shape
                x, y = int(lm.x*w), int(lm.y*h)

                self.landmark_list.append([id, x, y])


    def fingerUp(self):

        fingers = []
        if len(self.landmark_list) != 0:

            if self.landmark_list[self.tipIds[0]][1] < self.landmark_list[self.tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            for id in range(1, 5):

                if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2][2]:
                    fingers.append(1)

                else:
                    fingers.append(0)

        return fingers




def main():
    cap = cv2.VideoCapture(0)
    detector = simpleHandDetector()

    while True:
        isTrue, frame = cap.read()
        frame = cv2.flip(frame, 1)

        frame = detector.findHands(frame)
        f = detector.fingerUp()


        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xff == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()