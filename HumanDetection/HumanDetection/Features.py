# -*- coding: iso-8859-15 -*-

import os
import cv2
import random
import numpy as np

"""
Berechnet die HOGFeatures für positive Bilder. Hierbei MUSS das Bildformat der Patch-Größe entsprechen
"""
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


"""
Berechnet negative Features. Hierbei wird angegeben wie viele Patches aus einem Bild verwendet werden können.
Diese werden dann zufällig aus jedem Bild ausgeschnitten und die HOG Features berechnet
"""
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


"""
Öffnet ein Bild aus dem positiven Trainingsdaten
"""
def openPositiveImage(fullImgName):

    npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)

    return npImage


"""
Öffnet ein Bild aus dem Negativen Trainingsdaten und wählt ein zufälliges Patch aus
"""
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


"""
Berechnet die HOG Freatures für ein Patch
"""
def calcHog(image):
    
    hog = cv2.HOGDescriptor((image.shape[1],image.shape[0]), (16,16), (8,8), (8,8), 9) # weniger features kein prob mehr

    h = hog.compute(image) # hier Problem zu viele Features 94500 pro bild dauert 10minuten 450mb svm resultat
    
    return h

