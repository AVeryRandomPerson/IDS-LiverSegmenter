import numpy
import math
import colorsys


SAT = 1.0
VAL = 1.0

def _getGreyScaleIntensity( rgb):
    return math.floor(((rgb[0] + rgb[1] + rgb[2])/3))

def _computeColours( no_clusters):
    rgb_scheme = [[0,0,0],[255,255,255]]
    if(no_clusters > 2):
        colour_distributions = no_clusters-2
        for i in range(0, colour_distributions):
            hue_percentile = (1.0 / colour_distributions) * (i + 1.0)
            rgb = list(colorsys.hsv_to_rgb(hue_percentile, SAT, VAL))
            rgb[0] = math.ceil(rgb[0] * 255)
            rgb[1] = math.ceil(rgb[1] * 255)
            rgb[2] = math.ceil(rgb[2] * 255)
            rgb_scheme.append(rgb)

    return rgb_scheme

# Checks if pixel coordinates are near border.
def _isNearBorder( x, y, dimension):
    isNearLeftBorder = x <= 1
    isNearRightBorder = x >= (dimension[0] - 2)

    isNearTopBorder = y <= 1
    isNearBottomBorder = y >= (dimension[1] - 2)

    return (isNearLeftBorder or isNearRightBorder or isNearTopBorder or isNearBottomBorder)

def processAvgIntensityMap(image, width, height):
    average_intensity_map = numpy.zeros((width, height, 1))
    for y in range(0, height):
        for x in range(0, width):
            if _isNearBorder(x, y, (height, width)):
                average_intensity_map[y][x] = image[y][x]
            else:
                top = [image[y-2][x]]
                upper = [image[y-1][x-1],image[y-1][x],image[y-1][x+1]]
                mid = [image[y][x-2],image[y][x-1],image[y][x],image[y][x+1],image[y][x+2]]
                lower = [image[y+1][x-1],image[y+1][x],image[y+1][x+1]]
                btm = [image[y+2][x]]
                avg = math.floor(math.fsum(top+upper+mid+lower+btm))
                average_intensity_map[y][x] = avg

    return average_intensity_map

# Takes 2d_array, Returns 1d_numpy_array
def flatten1D(image):
    image_data = numpy.zeros(((len(image) ** 2), 1))
    for y in range(0, len(image)):
        for x in range(0, len(image[y])):
            image_data[(y * (len(image))) + x] = image[y][x]

    return image_data

def processClusteredImage(labels, no_clusters, width, height):
    rgb_scheme = _computeColours(no_clusters)
    processed_image = numpy.zeros((width, height, 3))
    x = 0
    y = 0
    for i in range(0, len(labels)):
        processed_image[y][x][0] = rgb_scheme[labels[i][0]][0]
        processed_image[y][x][1] = rgb_scheme[labels[i][0]][1]
        processed_image[y][x][2] = rgb_scheme[labels[i][0]][2]
        x = x + 1
        if (x == 512):
            x = 0
            y = y + 1

    return processed_image



