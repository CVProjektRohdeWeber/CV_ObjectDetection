import cv2
import numpy as np


def trainSVM(pos,neg):
    print 'start train method'

    amountPos = len(pos)
    amountNeg = len(neg)
    amount = amountPos + amountNeg
    
    trainData = np.concatenate((pos,neg))

    labels = np.zeros( amount, dtype = np.float32 )
    labels[:amountPos] = 1.   
    
    svm = cv2.SVM()

    #svm.train(trainData,responses, params=svm_params)
    #svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )

    svm_params = dict(kernel_type=cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC)

    svm.train_auto(trainData, labels, None, None, params=svm_params, k_fold=3) #kfold=3 (default: 10)

    print 'svm trained'

    return svm

def saveSVM(path,name, svm):
    
    print 'saving svm'
    svm.save(path+'\\'+name)
    print 'svm saved'
    return path+'\\'+name
