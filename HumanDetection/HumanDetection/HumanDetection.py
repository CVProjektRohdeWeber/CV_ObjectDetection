import sys
import os
import Features
import TrainSVM
import numpy as np
import cv2

if __name__ == '__main__':

    train = True
    test = True

    if len(sys.argv) > 1:
        path = sys.argv[1]

    currentFolder = os.getcwd() 

    if train == True:
       
        print "Start reading positive images"
        pos = Features.getPositiveFeatures(currentFolder+'\\posimg')

        print "Start reading negative images"
        size = [160,96] #x,y x=height y= width
        neg = Features.getRandomNegativeFeatures(currentFolder+'\\negimg',2,size)
  
        svm = TrainSVM.trainSVM(pos,neg)
        neg = None
        pos = None

        TrainSVM.saveSVM(currentFolder+'\\SVMs','human.xml', svm)

    if test == True:
        print 'test'


    #npImage = cv2.imread(currentFolder+'\\NegativeImages\\00000002a.png')
    #cv2.imshow( "Display window", npImage )  
    #cv2.waitKey(0) # Alex das hier war der fehler man muss waitkey benutzen damit das Bild angezeigt wird