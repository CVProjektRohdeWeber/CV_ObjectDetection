import cv2
import os
import numpy as np
import threading
import Queue


def loadSVM(path,name):

    print 'Start loading svm'
    svm = cv2.SVM()
    svm.load(path+'\\'+name)
    print 'svm loaded'

    return svm



def detectHumans(imgPath,svm,slidingWindowSize):
    print 'start bootstrapping'
    imageFiles = os.listdir(imgPath)
    detections = []
    threads = []
    returnqueue = Queue.Queue()

    for imgName in imageFiles:
        fullImgName = imgPath+'\\'+imgName
        npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)
        t = threading.Thread(None,detectThread,None,(npImage,svm,slidingWindowSize,returnqueue))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print str(returnqueue.unfinished_tasks)
    while returnqueue.unfinished_tasks>0:
        item = returnqueue.get()
        detections.append(item)
        returnqueue.task_done()


    return detections


    

def detectThread(imgResized,svm,slidingWindowSize,returnqueue):
    detections = []
    print "Thread started!"
    xMax = imgResized.shape[0] - slidingWindowSize[0]
    yMax = imgResized.shape[1] - slidingWindowSize[1]
    
    for i in range(0,xMax,10):
        for j in range(0,yMax,10):
            slide = imgResized[i:i+slidingWindowSize[0],j:j+slidingWindowSize[1]]
            
            hog = cv2.HOGDescriptor((slidingWindowSize[1],slidingWindowSize[0]), (16,16), (8,8), (8,8), 9)
            h = hog.compute(slide)

            detected = svm.predict(h,True)
            if detected < - 1.0:
                returnqueue.put(h)


