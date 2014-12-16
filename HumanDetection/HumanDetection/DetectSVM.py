# -*- coding: iso-8859-15 -*-

import cv2
import numpy as np
import threading
import Queue

"""
Laden und einbinden der SVM
"""
def loadSVM(path,name):

    print 'Start loading svm'
    svm = cv2.SVM()
    svm.load(path+'\\'+name)
    print 'svm loaded'

    return svm


"""
Hauptmethode der Personenerkennung

Funktionsweise:
Das übergebene Bild wird immer um 10% runterskaliert, bis es die Größe des sliding windows erreicht hat.
Für jede Skalierungsstufe wird ein Thread gestartet, der diese verarbeitet. Alle Treffer werden dabei zunächst
mit Position und Skalierungsfaktor in eine synchronisierte Queue gespeichert.

Diese Queue wird dannach in eine Liste umgewandelt, welche dann an die NMS übergeben werden kann
"""
def detectHumans(img,svm,slidingWindowSize):
    print 'start detect humans'

    detections = []
    threshold = 0.9
    factor = 1.0
    imgResized  = img
    threads = []
    returnqueue = Queue.Queue()

    while imgResized.shape[0] >= slidingWindowSize[0] and imgResized.shape[1] >= slidingWindowSize[1]:

        print imgResized.shape[0]
        print imgResized.shape[1]
        t = threading.Thread(None,detectThread,None,(imgResized,svm,slidingWindowSize,factor,returnqueue))
        threads.append(t)
        t.start()
        factor = factor-0.1
        imgResized  = cv2.resize(img,(0,0),fx = factor, fy=factor)
    for t in threads:
        t.join()

    while returnqueue.unfinished_tasks>0:
        item = returnqueue.get()
        detections.append(item)
        returnqueue.task_done()

    return detections


    
"""
Diese Methode wird jeweils von einem Thread gestartet und fährt das
Sliding Window über das skalierte Testbild.

Hierbei wird das Window jeweils um 10 Pixel verschoben.

An jeder Stelle werden die HOG Features des patches berechnet und dann mit der SVM abgeglichen.
Ist der Abstand zur Hyperebene über einer gesetzten Threshold, so wird es als Treffer gewertet.

Die so gefundenen Treffer werden dann alle in die oben genannte Queue weggeschrieben, 
bis das Bild vollständig untersucht ist

"""
def detectThread(imgResized,svm,slidingWindowSize,factor,returnqueue):
    detections = []
    print "Thread started!"
    xMax = imgResized.shape[0] - slidingWindowSize[0]
    yMax = imgResized.shape[1] - slidingWindowSize[1]
    
    for i in range(0,xMax,5):
        for j in range(0,yMax,5):
            slide = imgResized[i:i+slidingWindowSize[0],j:j+slidingWindowSize[1]]
            # Berechnen der HOG Features
            hog = cv2.HOGDescriptor((slidingWindowSize[1],slidingWindowSize[0]), (16,16), (8,8), (8,8), 9)
            h = hog.compute(slide)
            # Prüfen auf der SVM
            detected = svm.predict(h,True)
            detected = detected / factor
            #Prüfen der Threshold
            if detected < -1.0:
                rect = []
                rect[:] = j / factor , i / factor , (j+slidingWindowSize[1]) / factor, (i+slidingWindowSize[0]) / factor, factor, detected
                returnqueue.put(rect)
                print detected
                cv2.imshow("detect factor:"+str(factor),slide)
                cv2.waitKey(1)
    print "Thread done!"
    

#  Felzenszwalb et al. aber nach Adrian Rosebrock von http://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
def non_max_suppression_slow(detections, overlapThresh):
    print "start NMS"
    detections = np.array(detections)

    if detections.shape[0] == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = detections[:,1]
    y1 = detections[:,0]
    x2 = detections[:,3]
    y2 = detections[:,2]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in xrange(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    print "NMS done!"
    return detections[pick]

#  Felzenszwalb et al. https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m
def non_max_suppression_fast(detections, overlapThresh):

    detections = np.array(detections)

    if detections.shape[0] == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # jeweilige werte in eigene listen packen
    x1 = detections[:,1]
    y1 = detections[:,0]
    x2 = detections[:,3]
    y2 = detections[:,2]
    s = detections[:,5]
 
    #berechnung der flächeninhalte aller bounding boxen
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # sortieren der indizes nach dem score
    idxs = np.argsort(s)

    while len(idxs) > 0 :
        
      last = len(idxs) - 1
      i = idxs[last]
      pick.append(i)
      suppress = [last]

      for pos in xrange(0, last):
          j = idxs[pos]
          # boundingbox die um die andere bounding box liegt berechnen, für jede bounding box (i derzeitige bounding box, j alle anderen)
          xx1 = max(x1[i], x1[j])
          yy1 = max(y1[i], y1[j])
          xx2 = min(x2[i], x2[j])
          yy2 = min(y2[i], y2[j])
  
          # fläche der großen boundingbox berechnen
          w = xx2 - xx1 + 1
          h = yy2 - yy1 + 1
          
          if w > 0 and h > 0:
            #overlap der neuen bounding box berechnen
            overlap = float(w * h) / area[j]

            #wenn overlap groß genug dann wird es unterdrückt
            if overlap > overlapThresh:
                suppress.append(pos)
      idxs = np.delete(idxs, suppress)     #löschen aller boundingboxes die überdeckt werden
      
    top = detections[pick] #auswählen der boundingboxen deren index nicht unterdrückt wurde
    
    return top

# boundingboxen visualisieren ohne Skalierung
def vizualizeDetections2(img, detections):
    print 'start vizualization'

    for r in detections:
        rx = int(r[0] * r[4]) 
        ry = int(r[1] * r[4])
        rw = int(r[2] * r[4])
        rh = int(r[3] * r[4])
        tl = (rx + 10  , ry + 10)
        br = (rx + rw -10, ry + rh -10)
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)

    return img

# boundingboxen visualisieren mit Skalierung
def vizualizeDetections(img, detections):
    print 'start vizualization'

    for r in detections:
        rx = int(r[0])
        ry = int(r[1])
        rw = int(r[2])
        rh = int(r[3])
        tl = (rx + 10  , ry + 10)
        br = (rw -10, rh -10)
        cv2.rectangle(img, tl, br, (0, 255, 0), 2)

    return img


