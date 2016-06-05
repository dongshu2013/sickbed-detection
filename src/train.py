from utils import file_iterator, build_hog
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn import cross_validation
import numpy as np
import cv2

MODEL_PATH = '../model/svmModel.pkl'

def load_data(hog):
    filelist = {}
    for f in file_iterator('../data/rgb-image-train/rotated-positive', 'jpg'):
        filelist[f] = 1

    for f in file_iterator('../data/rgb-image-train/negative', 'jpg'):
        filelist[f] = 0

    files = filelist.keys()
    label = [filelist[f] for f in files]
    label = np.array(label).reshape(-1)

    data = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 128))
        data.append(hog.compute(img).reshape(-1))
    return data, label

def estimated_classifier(data, label):
    tuned_params = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 1000]}
    ]
    score = "f1"
    print("Tunning Parameters for %s".format(score))
    grid_search = GridSearchCV(LinearSVC(C=1, class_weight='balanced'), tuned_params, cv=5, scoring='%s' % score)
    grid_search.fit(data, label)
    print("Best parameters set found on development set:")
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search.best_estimator_

def train_clf(data, label):
    clf = estimated_classifier(data, label)
    clf.fit(data, label)
    joblib.dump(clf, MODEL_PATH)

if __name__ == '__main__':
    hog = build_hog(winSize=(128, 128))
    data, label = load_data(hog)
    train_clf(data, label)
