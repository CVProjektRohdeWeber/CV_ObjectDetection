import cv2
import os
import numpy as np
import threading
import Queue

"""
Diese Methode sucht alle positiven Features in den negativen Testdaten für das Hard Negative Mining.
Diese Daten werden dann zum Bootstrapping verwendet.

Die Funktion gleicht beinahe der normalen Detection, allerdings wird hier nicht skaliert und ein ganzer Ordner abgearbeitet.
Dabei wird für jedes Bild ein eigener Thread gestartet, welcher die False Negatives sucht und in einer Queue speichert.
"""
def detectFalsePositives(imgPath,svm,slidingWindowSize):
    print 'start bootstrapping'
    imageFiles = os.listdir(imgPath)
    detections = []
    threads = []
    returnqueue = Queue.Queue()

    for imgName in imageFiles:
        fullImgName = imgPath+'\\'+imgName
        #Bild laden und Thread dafür starten
        npImage = cv2.imread(fullImgName,cv2.CV_LOAD_IMAGE_COLOR)
        t = threading.Thread(None,detectThread,None,(npImage,svm,slidingWindowSize,returnqueue))
        threads.append(t)
        t.start()

    #Warten bis alle Threads beendet sind
    for t in threads:
        t.join()

    #Queue in Liste umwandeln
    while returnqueue.unfinished_tasks>0:
        item = returnqueue.get()
        detections.append(item)
        returnqueue.task_done()


    return detections


    
"""
Diese Methode wird als Thread für jedes Bild im angegebenen Ordner gestartet.

Diese führt die Detektion mit einem Sliding Window aus und sucht alle False Positives.

Dabei wird auch hier ein Threshold verwendet, damit auch wirklich deutliche False Positives
für das Bootstrapping verwendet werden und somit die Datenmenge geringer gehalten werden kann
für eine bessere SVM.
"""
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
            if detected < -1.0:
                returnqueue.put(h)


