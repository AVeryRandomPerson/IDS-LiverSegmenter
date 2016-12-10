import numpy
import os
import cv2
import random
from matplotlib import pyplot
#CONSTANTS
HEXTODEC= True
DECTOHEX = False

HEXTORGB = True
DECTORGB = False

CLUSTER_K_MEANS_INTENSITY = 0
CLUSTER_K_MEANS_TEXTURE_INTENSITY = 1


COMPRESS_TEXTURE_MODE_INTENSITY = 0

# Takes 2d_array into 1d_numpy_array
# Useful to convert cv2 read Images into 1d stacks for clustering K-Means
def _2dFlatten1D(cv2Image,image_data):
    for y in range(0,len(cv2Image)):
        for x in range (0,len(cv2Image[y])):
            image_data[(y*(len(cv2Image)))+x] = cv2Image[y][x]


    return image_data

# HEX to DEC to HEX converter (Depends on mode)
# Useful for colouring clusters in visualization.
def _HEXDECConverter(value,mode):
    # Conversion Mode is True for Hex to Dec if value is string. Or else, its false, which means, mode is Dec to Hex
    HEXDEC_DICT = {
                '0': 0,
                '1': 1,
                '2': 2,
                '3': 3,
                '4': 4,
                '5': 5,
                '6': 6,
                '7': 7,
                '8': 8,
                '9': 9,
                'A': 10,
                'B': 11,
                'C': 12,
                'D': 13,
                'E': 14,
                'F': 15,
                }

    DECHEX_DICT = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
                10: 'A',
                11: 'B',
                12: 'C',
                13: 'D',
                14: 'E',
                15: 'F',
                }

    if(mode):
        if(not type(value) == 'str'):
            str(value)

        value = value.upper()
        DEC = 0
        for i in range(0,len(value)):
            DEC = DEC + (HEXDEC_DICT[value[i]] * 16**(len(value)-i-1))

        return DEC

    else:
        HEX = [] #will use reverse before converting to str

        currentValue = value
        while(currentValue > 1):
            HEX.append(DECHEX_DICT[int(currentValue) % 16])
            currentValue = currentValue/16



        HEX.reverse()
        HEX = ''.join(HEX)
        return HEX

# COMPUTES RGB from HEX OR DEC (Depends on mode)
def _calcRGB(value,mode):
    if(mode):
        red = _HEXDECConverter(value[0:2],HEXTORGB)
        green = _HEXDECConverter(value[2:4],HEXTORGB)
        blue = _HEXDECConverter(value[4:6],HEXTORGB)

    else:
        HEX = _HEXDECConverter(value,DECTOHEX)
        red = _HEXDECConverter(HEX[0:2], HEXTORGB)
        green = _HEXDECConverter(HEX[2:4], HEXTORGB)
        blue = _HEXDECConverter(HEX[4:6], HEXTORGB)


    return [red, green, blue]

# Takes cluster results and visualizes a new 2d_array of image data. Displayable by cv2.imshow()
def _visualizeCluster(labels,image_dimension,no_clusters):
    #if cluster is 4 or less , we use predefined colours R G B Y
    if(no_clusters <= 4 ):
        cluster_colours_rgb = [[255,0,0],[0,255,0],[0,0,255],[255,255,0]]

    else:

        MAX = 16777215.0
        intervals = MAX/(no_clusters - 1)
        cluster_colours_dec = [0.0]
        cluster_colours_rgb = [[0,0,0]]


        for i in range(1, no_clusters):
            cluster_colours_dec.append(cluster_colours_dec[i - 1] + intervals)


        for i in range(1, no_clusters):
            cluster_colours_rgb.append(_calcRGB(cluster_colours_dec[i],DECTORGB))

    processed_image = numpy.zeros((image_dimension[0], image_dimension[1], 3))
    x = 0
    y = 0
    for i in range(0, len(labels)):
        processed_image[y][x][0] = cluster_colours_rgb[labels[i][0]][0]
        processed_image[y][x][1] = cluster_colours_rgb[labels[i][0]][1]
        processed_image[y][x][2] = cluster_colours_rgb[labels[i][0]][2]

        x = x + 1
        if (x == 512):
            x = 0
            y = y + 1


    return processed_image


# Checks if pixel coordinates are near border.
def _isNearBorder(x,y,dimension):
    isNearLeftBorder = x<=1
    isNearRightBorder = x>=(dimension[0]-2)

    isNearTopBorder = y<=1
    isNearBottomBorder = y>=(dimension[1]-2)

    return(isNearLeftBorder or isNearRightBorder or isNearTopBorder or isNearBottomBorder)

# Calculates Image Data for clustering. Default mode uses Intensity.
def _compressTextureData(cv2Image,mode,image_data):
    height = len(cv2Image)
    width = len(cv2Image[0])
    if(mode == COMPRESS_TEXTURE_MODE_INTENSITY):
        for y in range(0,height):
            for x in range(0,width):
                if _isNearBorder(x,y,(height,width)):
                    image_data[(y * width) + x] = cv2Image[y][x]
                else:

                    top = int(cv2Image[y-2][x])
                    upper_layer =  int(cv2Image[y-1][x]) + int(cv2Image[y-1][x-1]) + int(cv2Image[y-1][x+1])
                    mid_layer = int(cv2Image[y][x-2]) + int(cv2Image[y][x-1]) + int(cv2Image[y][x]) + int(cv2Image[y][x+1]) + int(cv2Image[y][x+2])
                    bottom_layer = int(cv2Image[y+1][x]) + int(cv2Image[y+1][x-1]) + int(cv2Image[y+1][x+1])
                    bottom = int(cv2Image[y+2][x])

                    average_intensity = (top + upper_layer + mid_layer + bottom_layer + bottom)/13
                    image_data[(y*(len(cv2Image)))+x] = int(average_intensity)
        return image_data

# image_path is a string. Full path using forward slash '/'
# cv2_kc_parameters is in the form of (number_clusters, criteria, attempts, flags)
# display is a boolean to display the image
def clusterImage(image_path, cv2_kc_parameters, write = False, mode = CLUSTER_K_MEANS_INTENSITY):
    cv2Image = cv2.imread(image_path, 0)
    image_dimension = (len(cv2Image), len(cv2Image[0]))

    image_data = numpy.zeros(((len(cv2Image)**2),1))
    if(mode == CLUSTER_K_MEANS_INTENSITY):
        image_data = _2dFlatten1D(cv2Image,image_data)

    elif(mode == CLUSTER_K_MEANS_TEXTURE_INTENSITY):
        image_data = _compressTextureData(cv2Image,COMPRESS_TEXTURE_MODE_INTENSITY,image_data)


    image_data = numpy.float32(numpy.array(image_data))
    compactness, labels, centers = cv2.kmeans(data=image_data,
                                              K=cv2_kc_parameters[0],
                                              bestLabels=None,
                                              criteria=cv2_kc_parameters[1],
                                              attempts=cv2_kc_parameters[2],
                                              flags=cv2_kc_parameters[3])
    processed_image = _visualizeCluster(labels, image_dimension,cv2_kc_parameters[0])

    if(write):
        cv2.imwrite("C:/Users/acer/Desktop/TestSamples/processed.jpg", processed_image)


    return compactness, labels, centers, processed_image

