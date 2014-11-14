import cv2
import numpy as np


def loadSVM(path,name):

    print 'Start loading svm'
    svm = cv2.SVM()
    svm.load(path+'\\'+name)
    print 'svm loaded'

    return svm