import numpy as np
import cv2


def draw_bounding_rect(image, landmarks_point):
    landmarks_point = [[int(n[0]*image.shape[1]),int(n[1]*image.shape[0])] for n in landmarks_point]

    x, y, w, h = cv2.boundingRect(np.array(landmarks_point))
    brect = [x, y, x + w, y + h]
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                  (50, 200, 0), 1)

    return brect
