import cv2
import numpy
import imageHandler

class KMeansClusterer():



    def __init__(self,location):
        self.image = cv2.imread(location , 0)
        self.width = len(self.image[0])
        self.height = len(self.image)
        self.image_dimension = (self.width,self.height)

    def clusterImageBasic(self, no_clusters, criteria, attempts, flags):
        image_data = imageHandler.flatten1D(self.image)
        image_data = numpy.float32(numpy.array(image_data))
        compactness, labels, centers = cv2.kmeans(data=image_data,
                                                  K=no_clusters,
                                                  bestLabels=None,
                                                  criteria=criteria,
                                                  attempts=attempts,
                                                  flags=flags)
        self.processed_image = imageHandler.processClusteredImage(labels, no_clusters, self.width, self.height)
        return compactness, labels, centers, self.processed_image

    def clusterImageAverageIntensity(self, no_clusters, criteria, attempts, flags):
        avgIntensityMap = imageHandler.processAvgIntensityMap(self.image, self.width, self.height)
        image_data = imageHandler.flatten1D(avgIntensityMap)
        image_data = numpy.float32(numpy.array(image_data))
        compactness, labels, centers = cv2.kmeans(data=image_data,
                                                  K=no_clusters,
                                                  bestLabels=None,
                                                  criteria=criteria,
                                                  attempts=attempts,
                                                  flags=flags)
        self.processed_image = imageHandler.processClusteredImage(labels, no_clusters, self.width, self.height)

        return compactness, labels, centers, self.processed_image

    def writeResult(self, write_file_path):
        cv2.imwrite(write_file_path, self.processed_image)