from fnmatch import fnmatch
import imutils
import time
import csv
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

def file_iterator(input_dir, extension):
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if fnmatch(f, '*.' + extension):
                yield os.path.join(root, f)

def build_hog(winSize=(128,128), blockSize=(8,8), blockStride=(4,4), cellSize=(4,4), nbins=9):
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    return cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

def sliding_window(image, stepSize, windowSize):
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            right = image.shape[1] -1 if x + windowSize[0] >= image.shape[1] else x + windowSize[0]
            bottom = image.shape[0] - 1 if y + windowSize[1] >= image.shape[0] else y + windowSize[1]
            yield (x, y, image[y:bottom, x:right])

def pyramid(image, scale=1.2, minSize=(128, 128)):
    width = image.shape[1]
    yield image.copy(), 1
    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=min(380, w))
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image.copy(), width / float(image.shape[1])

def load_bounding_box(filepath):
    bounding_box = {}
    csv_data = csv.reader(open(filepath))
    row_num = 0
    for row in csv_data:
        if row_num == 0:
            tags = row
        else:
            bounding_box[row[0]] = [int(r) for r in row[1:]]
        row_num = row_num + 1
    return bounding_box

def draw_rectangle(img, window, color=(0, 55, 255)):
    cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), color, +2, 4)

def overlap(label_box, pred_box):
    x1 = max(label_box[0], pred_box[0])
    y1 = max(label_box[1], pred_box[1])
    x2 = min(label_box[2], pred_box[2])
    y2 = min(label_box[3], pred_box[3])
    return [x1, y1, x2, y2]

def cpt_area(box):
    return (box[0] - box[2]) * (box[1] - box[3])

def timestamp():
    return int(round(time.time() * 1000))

def abs(val):
    return val if val > 0 else -val

def rotate(img, degree):
    w, h = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1)
    return cv2.warpAffine(img, M, (w, h))

def rotated():
    for f in file_iterator('../data/rgb-image-train/positive', 'jpg'):
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        for degree in range(0, 360, 90):
            cv2.imwrite('../data/rgb-image-train/rotated-positive/' + \
                    str(degree) + '_' + os.path.basename(f), \
                    rotate(img, degree))
