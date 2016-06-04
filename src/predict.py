from utils import file_iterator, pyramid, sliding_window, load_bounding_box, draw_rectangle
from sklearn.externals import joblib
import numpy as np
import csv
import cv2
import os

def build_bounding_box(pred_boxes):
    xu = sorted([p[0] for p in pred_boxes])[0]
    yu = sorted([p[1] for p in pred_boxes])[0]
    xb = sorted([p[2] for p in pred_boxes])[-1]
    yb = sorted([p[3] for p in pred_boxes])[-1]
    return xu, yu, xb, yb

def cpt_accuracy(label_box, pred_box):
    x1, y1, x2, y2 = overlap_area(label_box, pred_box)
    s1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    s2 = (label_box[2] - label_box[0]) * (label_box[3] - label_box[1])
    s_overlap = (x2 - x1) * (y2 - y1)
    return s_overlap/float(s1 + s2 - s_overlap)

def positive_windows(origin_img, hog, clf):
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
        w = np.concatenate((points, points + 128*scale), axis=1)
        windows = windows + w.tolist()
    return positive_windows

def get_corner(points):
    x1 = sorted([p[0] for p in points])[0]
    y1 = sorted([p[1] for p in points])[0]
    x2 = sorted([p[2] for p in points])[-1]
    y2 = sorted([p[2] for p in points])[-1]
    return [x1, y1, x2, y2]

def predict(img, label_box, hog, clf):
    windows = positive_windows(img, hog, clf)
    if len(windows) == 0:
        return img, None
    pred_box = get_corner(windows)
    return pred_box

def process(img_folder, label_boxes, hog, clf):
    accuracy = []
    for f in file_iterator(img_folder):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        label_box = label_boxes[os.path.basename(f)]
        print "Processing %s" % f
        pred_box = detect(img, hog, clf)

        accuracy = cpt_accuracy(label_box, pred_box)
        accuracy.append([os.path.basename(f), accuracy])
        if accuracy < 0.4:
            cv2.imwrite('./data/rgb-low-accuracy/' + os.path.basename(f), img)

        draw_rectange(img, pred_box, color=(0,55,255))
        draw_rectange(img, label_box, color=(255,55,0))
        cv2.imwrite('./predict/' + os.path.basename(f), img)

    with open("../accuracy.txt", "w+") as af:
        writer = csv.writer(af, delimiter=',')
        af.writerow(['filename', 'accurccy'])
        af.writerows(accuracy)

if __name__ == '__main__':
    hog = build_hog(winSize=(128, 128))
    clf = joblib.load('./models/svmModel.pkl')
    label_boxes = load_bounding_box('./bounding_box.csv')
    process('./test/', label_boxes, hog, clf)
