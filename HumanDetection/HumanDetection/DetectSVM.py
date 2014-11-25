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
    factor = 1.0
    imgResized  = img

    while imgResized.shape[0] >= slidingWindowSize[0] and imgResized.shape[1] >= slidingWindowSize[1]:

        print imgResized.shape[0]
        print imgResized.shape[1]

        xMax = imgResized.shape[0] - slidingWindowSize[0]
        yMax = imgResized.shape[1] - slidingWindowSize[1]
    
        for i in range(0,xMax,5):
            for j in range(0,yMax,5):
                slide = imgResized[i:i+slidingWindowSize[0],j:j+slidingWindowSize[1]]

                hog = cv2.HOGDescriptor((slidingWindowSize[1],slidingWindowSize[0]), (16,16), (8,8), (8,8), 9)
                h = hog.compute(slide)

                detected = svm.predict(h,True)
                detected = detected / factor
                if detected < - 1.5:
                    rect = []
                    rect[:] = j / factor, i / factor, j+slidingWindowSize[1] / factor, i+slidingWindowSize[0] / factor, factor, detected
                    detections.append(rect)
                    print detected
                    cv2.imshow("detect",slide)
                    cv2.waitKey(1)

        factor = factor-0.1
        imgResized  = cv2.resize(img,(0,0),fx = factor, fy=factor)

    return detections

def suppress(detections):
    new_detections = []
    
    detections.sort(key=lambda x: x[5], reverse=True)
    print detections
    new_detections.append(detections.pop())


    return new_detections


#  Felzenszwalb et al.
def non_max_suppression_slow(detections, overlapThresh):
    detections = np.array(detections)

    if detections.shape[0] == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = detections[:,0]
    y1 = detections[:,1]
    x2 = detections[:,2]
    y2 = detections[:,3]

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
    return detections[pick]

def detectHumansTest(img):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog.detectMultiScale(img, winStride=(8,8), padding=(16,16), scale=1.05)

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


