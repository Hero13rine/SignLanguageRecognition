import cv2
import numpy as np
import time
import os
import mediapipe as mp
from HandTrackingModule import handDetector


def get_fingerlocation(fpaths):
    data = []
    for path in fpaths:
        wCam, hCam = 640, 480
        pTime = 0
        cTime = 0
        cap = cv2.VideoCapture(path)
        cap.set(3, wCam)
        cap.set(4, hCam)
        """while True:
            success, frame = cap.read()
            if not success:
                break
            handdetector = handDetector()
            handdetector.findHands(frame)
            lmlist = handdetector.findPosition(frame)
            brect = handdetector.draw_bounding_rect(frame)
            # cv2.imshow("image", frame)
            # cv2.waitKey(1)
            data.append(lmlist)"""
        while True:
            success, img = cap.read()
            if success == False:
                break
            hand = handDetector()
            hand.findHands(img)
            lmlist = hand.findPosition(img)

            data.append(lmlist)
            """cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)
            cv2.imshow("image", img)
            cv2.waitKey(1)"""
        np.savez_compressed("./npz_files/" + label[i] + ".npz", data=data)
    return data


