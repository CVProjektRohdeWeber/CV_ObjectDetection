import sys
import os
import time
import cv2
import random
import numpy as np
import cv2.cv as cv


def getPositiveFeatures(path):

    imageFiles = os.listdir(path)
    features = []

    for imgName in imageFiles:    
            
        fullImgName = path+'\\'+imgName
        image = openPositiveImage(fullImgName)
        features.append(calcHog(image))

    return features


     
def getRandomNegativeFeatures(path, amount, size):

    imageFiles = os.listdir(path)
    features = []
    images = []

    for imgName in imageFiles:        
        fullImgName = path+'\\'+imgName
        images.append(openNegativeImage(fullImgName,size))
        images.append(openNegativeImage(fullImgName,size))
        for image in images:
            features.append(calcHog(image)) 

    return features



def openPositiveImage(fullImgName):

    npImage = cv2.imread(fullImgName,0)

    return npImage



def openNegativeImage(fullImgName, size):
    npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)
    imgX = npImage.shape[0]
    imgY = npImage.shape[1]  
    x = random.randint(0, imgX - size[0])
    y = random.randint(0, imgY - size[1])
    crop_img = npImage[x:y, size[0]:size[1]]
    return crop_img



def calcHog(image):

    hog = cv2.HOGDescriptor()
    h = hog.compute(image)

    return h

