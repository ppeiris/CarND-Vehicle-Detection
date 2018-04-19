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
from hog_featureextraction import *
from color_featureextraction import *
from windowsearch import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog


tosave = {}
"""
Compute color and hog feature vectors and combine them to get a one vector
create a training and test data sets
Normalize training and test data sets
"""
def getHogColorFeatures(features="HOG"):
    global tosave
    if features == 'HOG':
        hog_car_features, hog_notcar_features = extractHogFeatures()
        car_features = hog_car_features
        notcar_features = hog_notcar_features
    if features == 'COLOR':
        color_car_features, color_notcar_features = extractcolorfeatures()
        car_features = color_car_features
        notcar_features = color_notcar_features

    if features == 'BOTH':
        hog_car_features, hog_notcar_features = extractHogFeatures()
        color_car_features, color_notcar_features = extractcolorfeatures()
        print("combining both HOG and Color feature vectors to a one")
        car_features = np.hstack([color_car_features, hog_car_features]).astype(np.float64)
        notcar_features = np.hstack([color_notcar_features, hog_notcar_features]).astype(np.float64)

    # combine color and hog features to a one feature vector
    X = np.vstack([car_features, notcar_features]).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Split up data into randomized training and test sets

    print("Split features in to Training and Test data sets")
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    tosave["X_scaler"] = X_scaler
    tosave.update(getHogParams())
    tosave.update(getColorParams())
    # Normalized data
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    return X_train, y_train, X_test, y_test


""" return the SVM classifier

- load data training data
- build feature vectors
- train a classifier
- save trained classifier
- return the classifier

"""
def getClassifier():
    global tosave
    filename = "params.pkl"
    # Load the trained model from the disk if available
    if os.path.isfile(filename):
        print("Loading the Classifier %s from disk " %(filename))
        with open(filename, 'rb') as fid:
            paramsData = cPickle.load(fid)
        return paramsData

    # Train a model using features
    X_train, y_train, X_test, y_test = getHogColorFeatures('BOTH')
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
    tosave['svc'] = svc
    with open(filename, 'wb') as fid:
        cPickle.dump(tosave, fid)
    print("Final model has been saved as %s" %(filename))
    return tosave


def sligingWindow(params):

    # load a test image

    img = mpimg.imread('test_images/test1.jpg')

    ystart = 300
    ystop = 656
    scale = 1.5
    svc = params["svc"]
    X_scaler = params["X_scaler"]
    orient = params["orient"]
    pix_per_cell = params["pix_per_cell"]
    cell_per_block = params["cell_per_block"]
    spatial_size = params["spatial_size"]
    hist_bins = params["hist_bins"]

    out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

    return out_img



def main():
    params = getClassifier()
    sligingWindow(params)


if __name__ == '__main__':
    main()

