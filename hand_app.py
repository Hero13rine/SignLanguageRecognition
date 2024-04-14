import cv2
import time
import torch
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
import numpy as np
from Handmodel import HandModel
from tools.draw_landmarks import draw_landmarks
from tools.draw_bounding_rect import draw_bounding_rect
from tools.draw_rect_text import draw_rect_txt
from tools.landmark_handle import landmark_handle
import pyttsx3
import threading
from tools.Landmarks_relative_to_absolute import absolute_to_relative

from PIL import Image, ImageDraw, ImageFont

model_path = '11_16_param.pkl'  # 模型路径
# model_path = 'checkpoints/model_test1.pth' #模型保存路径
# label = [ "challenge", "cup","everyone", "fine","get", "here", "honor","I", "Love",  "very"]
label = [ "fine","fine","fine","fine","fine","fine","fine","fine","fine","fine"]
label_china = {"I":"我", "Love":"爱", "challenge":"挑战", "cup":"杯","everyone":"大家", "fine": "好"
               ,"get": "来到", "here": "这里", "honor":"荣幸", "very":"非常"}



label_num = len(label)
model = HandModel(label_num,batch_size=32)
model.eval()
model.load_state_dict(torch.load('11_16_param.pkl'))

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

FrameCount = 0
time1 = time.time()
fps = 0
background_flag = 0
background_color = 128
engine = pyttsx3.init()
# this_label='start'
start = time.perf_counter()
time_speech_epsode = 1.5

class test_dataset(Dataset):
    def __init__(self,data):
        # 遍历npz取数据
        self.data = data


    def test(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]



def run_speech(label):
    engine.say("{}".format(label))
    # engine.say('{}'.format(label_china[label]))
    engine.runAndWait()


def run_speech_china(label):
    engine.say('{}'.format(label_china[label]))
    engine.runAndWait()


thread_speech = threading.Thread(target=run_speech, args=('欢迎使用智手译平台',), name='speech')
thread_speech.start()


def cv2AddChineseText(img, text, position, textColor=(200, 100, 250), textSize=40):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)

    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")

    draw.text(position, text, textColor, font=fontStyle)

    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def run_thumber(cap, speech):
    global FrameCount
    global time1
    global fps
    global this_label
    global start
    global time_speech_epsode
    ret, frame = cap.read()  # 读取
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(frame, 1)  # 水平翻转
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hand_local = []
    if results.multi_hand_landmarks:
        for id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[id].classification[0].label  # 获得Label判断是哪几手
            index = results.multi_handedness[id].classification[0].index  # 获取左右手的索引号
            # hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_local_r = []  # 右手
            hand_local_l = []  # 左手

            for id, lm in enumerate(hand_landmarks.landmark):
                if hand_label == 'Right':
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

            brect = draw_bounding_rect(frame, hand_local_r)
            brect = draw_bounding_rect(frame, hand_local_l)

            hand_local_r = absolute_to_relative(hand_local_r, hand_local_r[0])
            hand_local_l = absolute_to_relative(hand_local_l, hand_local_l[0])
            # 先左后右
            hand_local_r.extend(hand_local_l)

            if background_flag:
                frame = np.zeros(frame.shape, np.uint8)
                frame.fill(128)

            # draw_landmarks(frame, hand_local)

            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)



            hand_local = hand_local_r


    if hand_local:
        data = torch.Tensor(hand_local)
        data = data.view(1, 42, 3)
        # print(1)
        output = model(data)
        # print(1)
        #output = model(hand_local).ravel()
        output = output[0].detach().numpy()
        value = np.max(output)
        index = np.where(output == value)[0][0]

        # index, value = output.topk(1)[1][0], output.topk(1)[0][0]
        print(index,value)
        this_label = label[int(index)]

        if value > 0.3:  # 在窗口中显示的字 图片 要显示的字 坐标 字体 大小 颜色 粗度
            if speech == 0:
                cv2.putText(frame,
                            this_label,
                            (300, 450),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (200, 100, 250),
                            3)
                cv2.putText(frame,
                            str(value),
                            (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (200, 100, 250),
                            3)
            else:

                frame = cv2AddChineseText(frame,
                                          label_china[this_label],
                                          (255, 400)
                                          )

            end = time.perf_counter()
            if speech == 0:
                if (end - start) > time_speech_epsode:
                    thread_speech = threading.Thread(target=run_speech, args=(this_label,), name='speech')

                    thread_speech.start()
                    start = time.perf_counter()
            else:
                if (end - start) > time_speech_epsode:
                    thread_speech = threading.Thread(target=run_speech, args=(label_china[this_label],), name='speech')

                    thread_speech.start()
                    start = time.perf_counter()

            # speak = win.Dispatch("SAPI.SpVoice")
            # speak.Speak("你好")

    """else:
        print("sb")"""
    time2 = time.time()
    FrameCount += 1
    if time2 - time1 >= 0.5:
        if FrameCount > 0:
            fps = round(FrameCount / (time2 - time1), 2)
            time1 = time.time()
            FrameCount = 0

    return frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()  # 读取
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)  # 水平翻转
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        hand_local = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # hand_landmarks = results.multi_hand_landmarks[0]
                # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    hand_local.append([lm.x, lm.y, lm.z])

                if background_flag:
                    frame = np.zeros(frame.shape, np.uint8)
                    frame.fill(128)

                # draw_landmarks(frame, hand_local)

                for handLms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                brect = draw_bounding_rect(frame, hand_local)

                hand_local = absolute_to_relative(hand_local, hand_local[0])

        if hand_local:
            # data = test_dataset(hand_local)
            # data = DataLoader(dataset=data, batch_size=32)
            data = torch.Tensor(hand_local)
            data = data.view(1, 21, 3)

            # print(1)
            output = model(data)
            # print(1)
            output = output[0].detach().numpy()
            value = np.max(output)
            index = np.where(output == value)[0][0]
            print(value, index)
