import cv2
import mediapipe as mp
import numpy as np
import time

class handDetector():
    def __init__(self, static_image_mode=False,
                 max_num_hands=2,
                 model_complexity=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(static_image_mode=static_image_mode,
                                        max_num_hands=max_num_hands,
                                        model_complexity=model_complexity,
                                        min_detection_confidence=min_detection_confidence,
                                        min_tracking_confidence=min_tracking_confidence)

    def findHands(self, img, is_draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if is_draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, handNo=0, is_draw=True):
        lmlist=[]
        relatively_lmlist=[]
        self.dilist=[]
        if self.results.multi_hand_landmarks:
            # myhand = self.results.multi_hand_landmarks[handNo]
            for id, myhand in enumerate(self.results.multi_hand_landmarks):
                label = self.results.multi_handedness[id].classification[0].label  # 获得Label判断是哪几手
                index = self.results.multi_handedness[id].classification[0].index  # 获取左右手的索引号
                # print(label, index)
                for id, lm in enumerate(myhand.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # print(id, cx, cy)
                    lmlist.append([id, cx, cy])
                    #relatively_lmlist.append([id, lm.x, lm.y])
                    self.dilist.append([cx, cy])
                    if is_draw and id == 0:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                        cv2.putText(img,
                                    str(label),
                                    (300, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (200, 100, 250),
                                    3)
                        cv2.putText(img,
                                    str(index),
                                    (350, 450),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1.5,
                                    (200, 100, 250),
                                    3)
        return lmlist

    def draw_bounding_rect(self, image):
        # landmarks_point = landmarks_point[:, 1:2]

        x, y, w, h = cv2.boundingRect(np.array(self.dilist))
        brect = [x, y, x + w, y + h]
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                    (50, 200, 0), 1)

        return brect

    def write_dataset(self):
        pass


data=[]
def main():
    try:
        pTime = 0
        cTime = 0
        cap = cv2.VideoCapture('E:\\Demo_Project\\SignLanguageRecognition\\applacation\\video\\大家\\30s-1.mp4')

        while True:
            success, img = cap.read()
            img = cv2.flip(img, 1)
            # img = cv2.imread('E:/Demo_Project/SignLanguageRecognition/applacation/img/subsample/call/1db2c669-d03c-40fa-a313-b28f1339d4e4.jpg')
            hand = handDetector()
            hand.findHands(img, is_draw=True)
            lmlist = hand.findPosition(img, is_draw=True)
            if len(lmlist) != 0:
                print(lmlist[0])
            # data.append(lmlist)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
            cv2.imshow("image", img)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print(data[0])
        np.savez('2', data=data)

if __name__ == "__main__":
    main()
