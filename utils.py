import glob
import sys
import os
import numpy as np
import matplotlib.image as mpimg
import cv2
import _pickle as cPickle

cars = []
noncars = []

def getPklFileName():
    return "params.pkl"

"""
HOG parameters
"""
def getParams():

    filename = getPklFileName()
    # Load the trained model from the disk if available
    if os.path.isfile(filename):
        print("Loading the Classifier %s from disk " %(filename))
        with open(filename, 'rb') as fid:
            paramsData = cPickle.load(fid)
        return paramsData
    else:
        return {
            # Hog params
            "colorspace" : 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb,
            "orient" : 11,
            "pix_per_cell" : 16,
            "cell_per_block" : 2,
            "hog_channel" : 'ALL', # Can be 0, 1, 2, or "ALL",
            "pixels_per_cell" : (16, 16),
            "cells_per_block" : (2, 2),
            "block_norm" : 'L2-Hys',
            "transform_sqrt" : True,
            "visualise" : False,
            "feature_vector" : True,

            # model
            'svc' : None,
            'X_scaler' : None,
        }

def imgParams():
    return {
        #
        "ystart" : 400,
        "ystop" : 656,
        "scale" : 1.5
    }

"""
load car images
"""
def loadCarImagePaths():
    global cars
    if len(cars) > 0:
        return cars

    path = "img/vehicles/"
    fdirs = ["GTI_Far", "GTI_Left", "GTI_MiddleClose", "GTI_Right", "KITTI_extracted"]

    for d in fdirs:
        cars += glob.glob(path + d + "/*.png")
    return cars

"""
load non-car-images
"""
def loadNonCarImagePaths():
    global noncars
    if len(noncars) > 0:
        return noncars

    path = "img/non-vehicles/"
    fdirs = ["Extras", "GTI"]

    for d in fdirs:
        noncars += glob.glob(path + d + "/*.png")
    return noncars

"""
load one car and none car random image and return them
"""
def getSampleImageSet():
    cars = loadCarImagePaths()
    noncars = loadNonCarImagePaths()
    return (mpimg.imread(cars[np.random.randint(0, len(cars))]),
        mpimg.imread(noncars[np.random.randint(0, len(noncars))]))


def getSampleImageSetGray():
    cars = loadCarImagePaths()
    noncars = loadNonCarImagePaths()

    car_gray = mpimg.imread(cars[np.random.randint(0, len(cars))])
    car_gray = cv2.cvtColor(car_gray, cv2.COLOR_RGB2GRAY)

    noncars_gray =  mpimg.imread(noncars[np.random.randint(0, len(noncars))])
    noncars_gray = cv2.cvtColor(noncars_gray, cv2.COLOR_RGB2GRAY)

    return car_gray, noncars_gray

"""
Load some car and none car images with hits hog transformation to print for the
report
"""
def getRandomHogImagesToPrint():

    images = []
    for i in range(0, 4):
        car_gray, non_car_gray = getSampleImageSetGray()
        features, hog_image = get_hog_features(car_gray)
        images += [{'img': car_gray, 'hog': hog_image}]
        features, hog_image = get_hog_features(non_car_gray)
        images += [{'img': non_car_gray, 'hog': hog_image}]

    return images

