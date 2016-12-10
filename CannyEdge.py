import os
import numpy
import cv2
from matplotlib import pyplot


PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/Edited-1014.jpg"
IMAGE = cv2.imread(PATH + FILENAME,0)


EDGE_IMAGE = cv2.Canny(IMAGE,255,255)


pyplot.subplot(121)
pyplot.imshow(EDGE_IMAGE,cmap = 'gray')
pyplot.show()