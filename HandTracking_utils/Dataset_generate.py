import os
import cv2
import numpy as np
from HandTracking_utils.HandTrackingModule import handDetector
from cv2_get_fingerlocation import get_fingerlocation
# 提取文件中视频标签
videoRootpaths = 'E:/Demo_Project/智手译/applacation/video/'
videoName = os.listdir(videoRootpaths)
videoFpaths = [videoRootpaths + f'{f}' for f in videoName]
labels = [label[:-6] for label in videoName]

# handDetector = handDetector()
print(videoName)
print(videoFpaths)



for label in videoName:
    data = []
    path = videoRootpaths + label
    print(path)
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
        lmlist, relatively_lmlist = hand.findPosition(img)

        data.append(lmlist)
        """cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)"""
    print("done",label)
    np.savez_compressed("./npz_files/" + label + ".npz", data=data)

