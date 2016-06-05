from utils import file_iterator, pyramid, sliding_window, load_bounding_box, draw_rectangle, build_hog, overlap, abs, timestamp, cpt_area
from sklearn.externals import joblib
import numpy as np
import imutils
import csv
import cv2
import os

ACCURACY_FILE = '../results/accuracy.txt'
HARD_NEGATIVE_PATH = '../results/rgb-hard-negative/'
PREDICT_PATH = '../results/predict/'
LOW_ACCURACY_IMG_PATH = '../results/rgb-low-accuracy/'
PROGRESS_FILE = './.progress'
MODEL_PATH = '../models/svmModel.pkl'

def flattern(windows):
    return [window for scaled_window in windows for window in scaled_window]

def get_corner(points):
    x1 = min([p[0] for p in points])
    y1 = min([p[1] for p in points])
    x2 = max([p[2] for p in points])
    y2 = max([p[3] for p in points])
    return [x1, y1, x2, y2]

def cpt_accuracy(label_box, pred_box):
    s1 = cpt_area(pred_box)
    s2 = cpt_area(label_box)
    s_overlap = cpt_area(overlap(label_box, pred_box))
    return s_overlap/float(s1 + s2 - s_overlap)

def detect(origin_img, hog, clf):
    windows = []
    for img, scale in pyramid(origin_img):
        points = []
        features = []
        for (x1, y1, window) in sliding_window(img, 8, (128, 128)):
            if window.shape[0] == 128 and window.shape[1] == 128:
                features.append(hog.compute(window).reshape(-1))
                points.append([x1, y1])

        if len(features) == 0:
            continue

        Y = clf.predict(features)
        points = np.asarray(points)[Y==1] * scale
        w = np.concatenate((points, points + 128*scale), axis=1).astype(int)
        if w.shape[0] > 0:
            windows.append(w.tolist())
    return windows

def save_img_with_box(filename, img, pred_box, label_box):
    draw_rectangle(img, pred_box, color=(0,55,255))
    draw_rectangle(img, label_box, color=(255,55,0))
    cv2.imwrite(PREDICT_PATH + filename, img)

def is_false_positive(label_box, window, threshold=0.65):
    s_window = cpt_area(window)
    s_overlap = cpt_area(overlap(label_box, window))
    r = s_overlap/float(s_window)
    return r < threshold

def apply_hard_negative(img, windows, label_box, stride=20):
    pos_windows = [w for w in windows if is_false_positive(label_box, w)]
    x_cur, y_cur = 0, 0
    for pw in pos_windows:
        if abs(pw[0] - x_cur) > stride and abs(pw[1] - y_cur) > stride:
            x_cur, y_cur = pw[0], pw[1]
            filename = str(timestamp()) + '_hard_neg.jpg'
            filepath = os.path.join(HARD_NEGATIVE_PATH, filename)
            subimg = img[pw[1]:pw[3], pw[0]:pw[2]]
            subimg = cv2.resize(subimg, (128, 128))
            cv2.imwrite(filepath, subimg)

def save_progress(filename):
    with open(PROGRESS_FILE, 'a+') as pf:
        pf.write(filename)

def load_progress():
    if os.path.isfile(PROGRESS_FILE):
        lines = [line.rstrip('\n') for line in open(PROGRESS_FILE)]
        return set(lines)
    return set([])

def prepare(progress):
    if len(progress) == 0 or not os.path.isfile(ACCURACY_FILE):
        with open(ACCURACY_FILE, 'w+') as af:
            af.write('filename,accurccy\n')

def process(img_folder, label_boxes, hog, clf):
    progress = load_progress()
    prepare(progress)

    for f in file_iterator(img_folder, 'jpg'):
        if os.path.basename(f) in progress:
            continue

        print 'Processing %s ... ' % os.path.basename(f)
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        f = os.path.basename(f)

        print '    Detecting object and build bounding box ... '
        windows = detect(img, hog, clf)
        if len(windows) == 0:
            print '        Object is not detected!'
            save_progress(f)
            continue
        pred_box = get_corner(flattern(windows))
        label_box = label_boxes[f]

        print '    Computing accuracy ... '
        accuracy = cpt_accuracy(label_box, pred_box)
        with open(ACCURACY_FILE, 'a') as af:
            af.write(f + ',' + str(accuracy) + '\n')

        if accuracy < 0.5:
            print '    Applying hard negative learning ... '
            cv2.imwrite(LOW_ACCURACY_IMG_PATH + os.path.basename(f), img)
            apply_hard_negative(img, flattern(windows), label_box)

        save_img_with_box(f, img, pred_box, label_box)
        save_progress(f)

if __name__ == '__main__':
    hog = build_hog(winSize=(128, 128))
    clf = joblib.load(MODEL_PATH)
    label_boxes = load_bounding_box('../data/bounding_box.csv')
    process('../data/rgb-image-test/', label_boxes, hog, clf)
