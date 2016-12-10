import numpy
import os
import cv2
import cKImage
from matplotlib import pyplot



PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/I0001014.jpg"
IMAGE = cv2.imread(PATH + FILENAME,0)

# Algorithm stops every 10 iterations or if epsilon accuracy = 1.0
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ATTEMPTS = 10
FLAGS = cv2.KMEANS_RANDOM_CENTERS
CLUSTERS = 10

compactness, labels, centers, processed_image = cKImage.clusterImage(PATH+FILENAME, (CLUSTERS,CRITERIA,ATTEMPTS,FLAGS), True, cKImage.CLUSTER_K_MEANS_TEXTURE_INTENSITY)

cv2.imshow("Processed Clusters", cv2.imread("C:/Users/acer/Desktop/TestSamples/processed.jpg",cv2.IMREAD_COLOR))
cv2.waitKey(0)
cv2.destroyAllWindows()