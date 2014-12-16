# -*- coding: iso-8859-15 -*-

import sys
import os
import Features
import TrainSVM
import DetectSVM
import Bootstrapping
import numpy as np
import cv2

if __name__ == '__main__':

    size = [160,96] #x,y x=height y= width
    """
    Auswahl der passenden Funktionen, die von der Anwendung ausgef�hrt werden sollen.

    Hinweis: Damit bootstrapping funktionieren kann, muss vorher einmal ohne eine SVM trainiert werden!
    """
    train = False
    bootstrapping = False
    detect = True

    if len(sys.argv) > 1:
        path = sys.argv[1]

    currentFolder = os.getcwd() 

    if train == True:

        if bootstrapping == True:
            # F�hre Bootstrapping auf den negativen Trainingsdaten aus
            svm = DetectSVM.loadSVM(currentFolder +'\\SVMs','human.xml')
            falsePositives = Bootstrapping.detectFalsePositives(currentFolder +'\\NegativeImages',svm,size)
       
        print "Start reading positive images"
        pos = Features.getPositiveFeatures(currentFolder+'\\PositiveImages')

        print "Start reading negative images"
        
        neg = Features.getRandomNegativeFeatures(currentFolder+'\\NegativeImages',2,size)
        if bootstrapping == True:
            #wenn bootstrapping aktiv ist, dann f�ge die false positives an die liste der negativen features
            np.concatenate((neg,falsePositives))
        svm = TrainSVM.trainSVM(pos,neg)
        neg = None
        pos = None

        TrainSVM.saveSVM(currentFolder+'\\SVMs','human.xml', svm)

    if detect == True:
        print 'Start detecting'
        #Laden der SVM
        svm = DetectSVM.loadSVM(currentFolder +'\\SVMs','human.xml')
        #Einlesen des Testbildes
        npImage = cv2.imread(currentFolder +'\\testImages\\1.jpeg',cv2.CV_LOAD_IMAGE_COLOR)
        #Finden von Menschen mit der SVM
        detections = DetectSVM.detectHumans(npImage,svm,size)
        #NMS durchf�hren und dannach die ergebnisse Anzeigen lassen
        detections = DetectSVM.non_max_suppression_slow(detections,0.4)
        #detections = DetectSVM.non_max_suppression_fast(detections,0.4)

        img = DetectSVM.vizualizeDetections(npImage,detections)
        
        cv2.imshow( "Display window", img )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
