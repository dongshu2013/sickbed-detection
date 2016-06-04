#!/usr/bin/env python

from utils import file_iterator
import numpy as np
import random
import csv
import imutils
import cv2
import os


def video_to_frame(video_path, img_path, stride=200):
    for video_file in file_iterator('./video/depth/', 'm4v'):
        cap = cv2.VideoCapture(video_file)
        ret = False
        if cap.isOpened():
            ret, frame = cap.read()

        i = 0
        while(ret):
            i = i + 1
            ret, frame = cap.read()
            filename = os.path.basename(video_file) + '_' + str(i) + '.jpg'
            if i % stride == 0:
                cv2.imwrite(os.path.join(img_path, filename), frame)

        cap.release()
        cv2.destroyAllWindows()
