from utils import pyramid, sliding_window, load_bounding_box, file_iterator
from predict import positive_windows
from sklearn.externals import joblib
import time
import cv2
import os

def timestamp():
    return int(round(time.time() * 1000))

def is_false_positive(label_box, window, threshold=0.6):
    x1, y1, x2, y2 = overlap_area(label_box, window)
    s_window = (window[2] - window[0]) * (window[3] - window[1])
    s_overlap = (x2 - x1) * (y2 - y1)
    r = s_overlap/float(s_window)
    return r < threshold


def gather_false_positive_windows(img, label_box, hog, clf):
    for window in positive_windows(img, hog, clf):
        if is_false_positive(label_box, window):
            target_path = '../data/rgb-hard-negative/'
            filename = str(timestamp()) + '_generated_from_' + os.path.basename(f)
            cv2.imwrite(os.path.join(target_path, filename), img)

def process(img_folder):
    for f in file_iterator(img_folder):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        label_box = label_boxes[os.path.basename(f)]
        gather_false_positive_windows(img, label_box, hog, clf)

if __name__ == '__main__':
    hog = build_hog(winSize=(128, 128))
    clf = joblib.load('./models/svmModel.pkl')
    label_boxes = load_bounding_box('./bounding_box.csv')
    process('./data/rgb-low-accuracy')
