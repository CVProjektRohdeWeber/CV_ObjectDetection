import cv2
import numpy as np


def loadSVM(path,name):

    print 'Start loading svm'
    svm = cv2.SVM()
    svm.load(path+'\\'+name)
    print 'svm loaded'

    return svm



def detectHumans(img,svm,slidingWindowSize):
    print 'start detect humans'

    detections = []

    

    img  = cv2.resize(img,(0,0),fx = 0.5, fy=0.5)
    cv2.imshow("kleiner",img)
    cv2.waitKey(0)
    xMax = img.shape[0] - slidingWindowSize[0]
    yMax = img.shape[1] - slidingWindowSize[1]

    for i in range(0,xMax,10):
        for j in range(0,yMax,10):
            slide = img[i:i+slidingWindowSize[0],j:j+slidingWindowSize[1]]

            hog = cv2.HOGDescriptor((slidingWindowSize[1],slidingWindowSize[0]), (16,16), (8,8), (8,8), 9) # weniger features kein prob mehr
            h = hog.compute(slide)

            detected = svm.predict(h,True)
            
            if detected < - 1.0:
                rect = []
                rect[:] = j,i, slidingWindowSize[1], slidingWindowSize[0]
                detections.append(rect)
                print detected
                cv2.imshow("detect",slide)
                cv2.waitKey(1)


    return detections


def vizualizeDetections(img, detections):
    print 'start vizualization'

    for r in detections:
        rx = r[0]
        ry = r[1]
        rw = r[2]
        rh = r[3]
        tl = (rx + int(rw*0.1), ry + int(rh*0.07))
        br = (rx + int(rw*0.9), ry + int(rh*0.87))
        cv2.rectangle(img, tl, br, (0, 255, 0), 3)


    return img


