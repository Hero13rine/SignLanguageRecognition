import os

def get_data_path(Fpath, Access='img'):
    if Access == 'video':
        video_num = 1
        videoRootpaths = Fpath
        videoName = os.listdir(videoRootpaths)
        videoFpaths = []
        labels = os.listdir(videoRootpaths)
        for label in labels:
            video = []
            file_names = os.listdir(videoRootpaths + label)

            for file_name in file_names:
                videoa = os.path.join(videoRootpaths + label, file_name)
                video.append(videoa)
            videoFpaths.append(video)

        label_num = len(labels)
        print("label_num:" + str(label_num))

        return zip(labels, videoFpaths)

    elif Access == 'img':
        img_num = 1
        imgRootpaths = Fpath # 根目录
        imgFpaths = [] # 每个图片的位置路径
        labels = os.listdir(imgRootpaths)
        for label in labels:
            img = []
            file_names = os.listdir(imgRootpaths + label)

            for file_name in file_names:
                imga = os.path.join(imgRootpaths + label, file_name)
                img.append(imga)
            imgFpaths.append(img)

        return zip(labels, imgFpaths)

    elif Access == 'train':
        trainRootPath = Fpath
        npzName = os.listdir(trainRootPath)
        label = [n[:-6] for n in npzName]
        trainDataPath = [trainRootPath + n for n in npzName]

        return zip(label, trainDataPath)