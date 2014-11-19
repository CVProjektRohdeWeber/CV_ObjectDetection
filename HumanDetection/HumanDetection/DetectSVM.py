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

    x = random.randint(0, imgX - size[0])
    y = random.randint(0, imgY - size[1])

    for(int i=0; i < h; ++i)
        for(int j=0; j < w; ++j)
            hog = cv2.HOGDescriptor((slidingWindowSize.shape[1],slidingWindowSize.shape[0]), (16,16), (8,8), (8,8), 9) # weniger features kein prob mehr
            h = hog.compute(slide)

            detected = svm.predict(h)
            if detected == 1:
                detections[i,0] = x
                detections[i,1] = y
                detections[i,2] = w
                detections[i,3] = h 

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


