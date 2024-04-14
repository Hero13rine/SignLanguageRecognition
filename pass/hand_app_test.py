with mp_hands.Hands(
        static_image_mode=True,  # False表示为图像检测模式
        max_num_hands=2,  # 最大可检测到两只手掌
        model_complexity=0,  # 可设为0或者1，主要跟模型复杂度有关
        min_detection_confidence=0.5,  # 最大检测阈值
) as hands:
    for idx, file in enumerate(IMAGE_List):
        image = cv2.flip(cv2.imread(file), 1)  # 读取图片
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将其标为RGB格式
        t0 = time.time()
        results = hands.process(image)  # 使用API处理图像图像
        '''
        results.multi_handedness
        包括label和score,label是字符串"Left"或"Right",score是置信度
        results.multi_hand_landmarks
        results.multi_hand_landmrks:被检测/跟踪的手的集合
        其中每只手被表示为21个手部地标的列表,每个地标由x、y和z组成。
        x和y分别由图像的宽度和高度归一化为[0.0,1.0]。Z表示地标深度
        以手腕深度为原点，值越小，地标离相机越近。 
        z的大小与x的大小大致相同。
        '''
        t1 = time.time()
        fps = 1 / (t1 - t0)  # 实时帧率
        # print('++++++++++++++fps',fps)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 将图像变回BGR形式
        dict_handnumber = {}  # 创建一个字典。保存左右手的手势情况
        if results.multi_handedness:  # 判断是否检测到手掌
            if len(results.multi_handedness) == 2:  # 如果检测到两只手
                for i in range(len(results.multi_handedness)):
                    label = results.multi_handedness[i].classification[0].label  # 获得Label判断是哪几手
                    index = results.multi_handedness[i].classification[0].index  # 获取左右手的索引号
                    hand_landmarks = results.multi_hand_landmarks[index]  # 根据相应的索引号获取xyz值
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,  # 用于指定地标如何在图中连接。
                        mp_drawing_styles.get_default_hand_landmarks_style(),  # 如果设置为None.则不会在图上标出关键点
                        mp_drawing_styles.get_default_hand_connections_style())  # 关键点的连接风格
                    gesresult = ges(hand_landmarks)  # 传入21个关键点集合，返回数字
                    dict_handnumber[label] = gesresult  # 与对应的手进行保存为字典
            else:  # 如果仅检测到一只手
                label = results.multi_handedness[0].classification[0].label  # 获得Label判断是哪几手
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,  # 用于指定地标如何在图中连接。
                    mp_drawing_styles.get_default_hand_landmarks_style(),  # 如果设置为None.则不会在图上标出关键点
                    mp_drawing_styles.get_default_hand_connections_style())  # 关键点的连接风格
                gesresult = ges(hand_landmarks)  # 传入21个关键点集合，返回数字
                dict_handnumber[label] = gesresult  # 与对应的手进行保存为字典
        if len(dict_handnumber) == 2:  # 如果有两只手，则进入
            # print(dict_handnumber)
            leftnumber = dict_handnumber['Right']
            rightnumber = dict_handnumber['Left']
            '''
            显示实时帧率，右手值，左手值，相加值
            '''
            s = 'FPS:{0}\nRighthand Value:{1}\nLefthand Value:{2}\nAdd is:{3}'.format(int(fps), rightnumber, leftnumber,
                                                                                      str(leftnumber + rightnumber))  # 图像上的文字内容
        elif len(dict_handnumber) == 1:  # 如果仅有一只手则进入
            labelvalue = list(dict_handnumber.keys())[0]  # 判断检测到的是哪只手
            if labelvalue == 'Right':  # 左手,不知为何，模型总是将左右手搞反，则引入人工代码纠正
                number = list(dict_handnumber.values())[0]
                s = 'FPS:{0}\nRighthand Value:{1}\nLefthand Value:0\nAdd is:{2}'.format(int(fps), number, number)
            else:  # 右手
                number = list(dict_handnumber.values())[0]
                s = 'FPS:{0}\nLefthand Value:{1}\nRighthand Value:0\nAdd is:{2}'.format(int(fps), number, number)
        else:  # 如果没有检测到则只显示帧率
            s = 'FPS:{0}\n'.format(int(fps))

        y0, dy = 50, 25  # 文字放置初始坐标
        # image = cv2.flip(image,1) # 反转图像
        for i, txt in enumerate(s.split('\n')):  # 根据\n来竖向排列文字
            y = y0 + i * dy
            cv2.putText(image, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow('MediaPipe Gesture Recognition', image)  # 显示图像
        cv2.imwrite('save/{0}.jpg'.format(idx), image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

