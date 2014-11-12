import sys
import os
import Features
import numpy as np
import cv2

if __name__ == '__main__':

  if len(sys.argv) > 1:
    path = sys.argv[1]

  currentFolder = os.getcwd() 
  print "Start reading positive images"
  #Features.getPositiveFeatures(currentFolder+'\\PositiveImages')
  print "Start reading negative images"
  size = [96,160]
  #Features.getRandomNegativeFeatures(currentFolder+'\\NegativeImages',2,size)
  npImage = cv2.imread(currentFolder+'\\NegativeImages\\00000002a.png')
  cv2.imshow( "Display window", npImage )