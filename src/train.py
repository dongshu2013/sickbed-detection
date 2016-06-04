from utils import file_iterator, build_hog
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from sklearn import cross_validation

def load_data(filepath, hog):
    filelist = {}
    for f in file_iterator('../data/rgb-image-train/positive'):
        filelist[f] = 1

    for f in file_iterator('../data/rgb-image-train/negative'):
        filelist[f] = 0

    for f in file_iterator('../data/rgb-image-train/hard-negative'):
        filelist[f] = 0

    label = [filelist[f] for f in files]
    label = np.array(label).reshape(-1)

    data = []
    for f in filelist:
        img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
        data.append(hog.compute(img).reshape(-1))
    return data, label

def estimated_classifier(data, label):
    tuned_params = [
        {'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'C': [0.01, 0.1, 1, 10, 100, 1000, 10000]}
    ]
    score = "f1"
    print "Tunning Parameters for %s" % score
    grid_search = GridSearchCV(SVC(kernel='rbf', C=1, gamma=1e-4, class_weight='balanced'), tuned_params, cv=5, scoring='%s' % score)
    grid_search.fit(data, label)
    print "Best parameters set found on development set:"
    print grid_search.best_params_
    print grid_search.best_score_
    return grid_search.best_estimator_

def train_clf(data, label):
    clf = estimated_classifier(data, label)
    clf.fit(data, label)
    joblib.dump(clf, './models/svmModel.pkl')

if __name__ == '__main__':
    hog = build_hog(winSize=(128, 128))
    data, label = load_data(hog)
    clf = train_clf(data, label)
