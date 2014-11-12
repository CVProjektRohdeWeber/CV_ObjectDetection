import sys
import os
import time
import cv2
import random
import numpy as np


def getPositiveFeatures(path):

    imageFiles = os.listdir(path)
    features = []
    count = 0

    for imgName in imageFiles:    
            
        fullImgName = path+'\\'+imgName
        image = openPositiveImage(fullImgName)
        features.append(calcHog(image))
        count = count +1
        if count % 100 == 0:
            print "Progress: "+ str(count)

    print "Features Calculated"

    return features


     
def getRandomNegativeFeatures(path, amount, size):

    imageFiles = os.listdir(path)
    features = []
    count = 0

    for imgName in imageFiles:        
        fullImgName = path+'\\'+imgName
        for i in range(amount):
            image = openNegativeImage(fullImgName,size)
            features.append(calcHog(image))
            count = count +1

        if count % 100 == 0:
            print "Progress: "+ str(count)

    print "Features Calculated"

    return features



def openPositiveImage(fullImgName):

    npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)

    return npImage



def openNegativeImage(fullImgName, size):
    
    npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)
    imgX = npImage.shape[0]
    imgY = npImage.shape[1]  

    x = random.randint(0, imgX - size[0])
    y = random.randint(0, imgY - size[1])

    height = x + size[0]
    width = y + size[1]
    
    crop_img = npImage[ x:height, y:width ] #hier war der fehler

    return crop_img



def calcHog(image):

    hog = cv2.HOGDescriptor()
    h = hog.compute(image)

    return h

