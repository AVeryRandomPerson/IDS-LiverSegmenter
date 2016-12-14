import cv2
from KMeansClusterer import KMeansClusterer



PATH = "C:/Users/acer/Desktop/TestSamples"
FILENAME = "/I0001014.jpg"

# Algorithm stops every 10 iterations or if epsilon accuracy = 1.0
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ATTEMPTS = 10
FLAGS = cv2.KMEANS_RANDOM_CENTERS
CLUSTERS = 10

CLUSTERER = KMeansClusterer(PATH+FILENAME)
CLUSTERER.clusterImageAverageIntensity(CLUSTERS,CRITERIA,ATTEMPTS,FLAGS)
CLUSTERER.writeResult("C:/Users/acer/Desktop/TestSamples/I0001014-RESULTS.png")
cv2.imshow("Processed Clusters", cv2.imread("C:/Users/acer/Desktop/TestSamples/I0001014-RESULTS.png",cv2.IMREAD_COLOR))
cv2.waitKey(0)
cv2.destroyAllWindows()