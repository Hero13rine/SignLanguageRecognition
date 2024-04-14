import cv2
import mediapipe as mp
import numpy as np
from tools.landmark_handle import landmark_handle
import os
from tools.Landmarks_relative_to_absolute import absolute_to_relative
from get_data_path import get_data_path


Access = 'video'

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

rootFpath = 'E:/Demo_Project/SignLanguageRecognition/applacation/video/cut/'
# rootFpath = 'D:/test/'
Fpath = get_data_path(rootFpath, Access=Access)
for path in Fpath:
    label, labelFpath = path[0], path[1]
    print("label: ", label)

    data = []
    for name in labelFpath:
        # 图片转化
        if Access == 'img':
            for i in range(2):
                frame = cv2.imread(name)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if i == 1:
                    frame = cv2.flip(frame, 1)
                results = hands.process(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks:
                    hand_local = []
                    for id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        label = results.multi_handedness[id].classification[0].label  # 获得Label判断是哪几手
                        index = results.multi_handedness[id].classification[0].index  # 获取左右手的索引号
                        # hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        hand_local_r = [] # 右手
                        hand_local_l = [] # 左手

                        for id, lm in enumerate(hand_landmarks.landmark):
                            if label == 'Right':
                                hand_local_r.append([lm.x, lm.y, lm.z])
                            else:
                                hand_local_l.append([lm.x, lm.y, lm.z])

                        # 补充
                        if len(hand_local_r) == 0:
                            hand_local_r = np.zeros((21,3))
                            hand_local_r = hand_local_r.tolist()
                            if len(hand_local_l) == 0:
                                break
                        if len(hand_local_l) == 0:
                            hand_local_l = np.zeros((21, 3))
                            hand_local_l = hand_local_l.tolist()

                        # 转化相对坐标
                        hand_local_r = absolute_to_relative(hand_local_r, hand_local_r[0])
                        hand_local_l = absolute_to_relative(hand_local_l, hand_local_l[0])
                        # 先左后右
                        hand_local_r.extend(hand_local_l)
                        data.append(hand_local_r)
                    print(np.shape(data))
        # 视频转化
        elif Access == 'video':
            cap = cv2.VideoCapture(name)
            for i in range(2):
                ret, frame = cap.read()
                while ret is True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if i == 1:
                        frame = cv2.flip(frame, 1)
                    results = hands.process(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    if results.multi_hand_landmarks:
                        for id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            label = results.multi_handedness[id].classification[0].label  # 获得Label判断是哪几手
                            index = results.multi_handedness[id].classification[0].index  # 获取左右手的索引号
                            # hand_landmarks = results.multi_hand_landmarks[0]
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            hand_local_r = []  # 右手
                            hand_local_l = []  # 左手

                            for id, lm in enumerate(hand_landmarks.landmark):
                                if label == 'Right':
                                    hand_local_r.append([lm.x, lm.y, lm.z])
                                else:
                                    hand_local_l.append([lm.x, lm.y, lm.z])

                            # 补充
                            if len(hand_local_r) == 0:
                                hand_local_r = np.zeros((21, 3))
                                hand_local_r = hand_local_r.tolist()
                            if len(hand_local_l) == 0:
                                hand_local_l = np.zeros((21, 3))
                                hand_local_l = hand_local_l.tolist()

                            # 转化相对坐标
                            hand_local_r = absolute_to_relative(hand_local_r, hand_local_r[0])
                            hand_local_l = absolute_to_relative(hand_local_l, hand_local_l[0])
                            # 先左后右
                            hand_local_r.extend(hand_local_l)
                            data.append(hand_local_r)
                    ret, frame = cap.read()

    if len(data) != 0:
        print(data[23])
    np.savez_compressed("./npz_files/" + path[0] + "_1.npz", data=data)

