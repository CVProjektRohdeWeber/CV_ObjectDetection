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

        npImage = cv2.imread(currentFolder +'\\testImages\\landschaftmensch2.png',cv2.CV_LOAD_IMAGE_COLOR)
        #npImage = cv2.imread(currentFolder +'\\testImages\\test.jpg',cv2.CV_LOAD_IMAGE_COLOR)
        
        #detections = DetectSVM.detectHumansTest(npImage)
        #img = DetectSVM.vizualizeDetections2(npImage,detections)
        
        detections = DetectSVM.detectHumans(npImage,svm,size)

        detections = DetectSVM.non_max_suppression_slow(detections,0.5)

        img = DetectSVM.vizualizeDetections(npImage,detections)
        
        cv2.imshow( "Display window", img )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
