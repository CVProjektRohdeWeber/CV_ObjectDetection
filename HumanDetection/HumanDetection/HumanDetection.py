import sys
import os
import Features
import TrainSVM
import DetectSVM
import numpy as np
import cv2

if __name__ == '__main__':

    size = [160,96] #x,y x=height y= width
    train = False
    detect = True


    if len(sys.argv) > 1:
        path = sys.argv[1]

    currentFolder = os.getcwd() 

    if train == True:
       
        print "Start reading positive images"
        pos = Features.getPositiveFeatures(currentFolder+'\\posimg')

        print "Start reading negative images"
        
        neg = Features.getRandomNegativeFeatures(currentFolder+'\\negimg',2,size)
  
        svm = TrainSVM.trainSVM(pos,neg)
        neg = None
        pos = None

        TrainSVM.saveSVM(currentFolder+'\\SVMs','human.xml', svm)

    if detect == True:
        print 'Start detecting'
        svm = DetectSVM.loadSVM(currentFolder +'\\SVMs','human.xml')

        npImage = cv2.imread(currentFolder +'\\testImages\\test.jpg',cv2.CV_LOAD_IMAGE_COLOR)

        detections = DetectSVM.detectHumans(npImage,svm,size)

        img = DetectSVM.vizualizeDetections(npImage,detections)
        
        cv2.imshow( "Display window", img )
        cv2.waitKey(0)

         

    #npImage = cv2.imread(currentFolder+'\\NegativeImages\\00000002a.png')
    #cv2.imshow( "Display window", npImage )  
    #cv2.waitKey(0) # Alex das hier war der fehler man muss waitkey benutzen damit das Bild angezeigt wird