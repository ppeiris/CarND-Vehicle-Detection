################################################################################
## Advanced Lane Finding Project
## Prabath Peiris
## peiris.prabath@gmail.com
################################################################################

import numpy as np
import cv2
import glob
import matplotlib
import time
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import os
import _pickle as cPickle
# import matplotlib.image as mpimg
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from utils import *
from featureextraction import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Parameters to tune
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"

""" HOG features

Extract HOG features for cars and non car images
"""
def extractHogFeatures():
    print("Extracting HOG features...")
    cars = loadCarImagePaths()
    cars = cars[0:500] # for testing

    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)


    notcars = loadNonCarImagePaths()
    notcars = notcars[0:500] # for testing

    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)

    print(round(time.time()-t, 2), 'Seconds to extract HOG features...')
    return car_features, notcar_features

def getLabledData():
    car_features, notcar_features = extractHogFeatures()
    # print((car_features))
    # print((notcar_features))

    features_ = []
    features_.append(car_features)
    features_.append(notcar_features)
    # Create an array stack of feature vectors
    X = np.vstack(features_).astype(np.float64)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)


    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)

    # Normalized data
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    return X_train, y_train, X_test, y_test

def buildClassifier():
    X_train, y_train, X_test, y_test = getLabledData()
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    print("Training a classifier...")
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


if __name__ == '__main__':
    buildClassifier()
    # processVideo()
