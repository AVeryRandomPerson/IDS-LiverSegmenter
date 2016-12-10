import dicom
import os
import numpy
from matplotlib import pyplot, cm


BASE_LOCATION = 'D:/Jon\'s Projects/UNMC/Year 3/INDIVIDUAL-DISSERTATION-FYP/Sample-CTs'
PATIENT_FOLDER = '/55144/'


PathDicom = BASE_LOCATION + PATIENT_FOLDER
DCM_FILES = []  # create an empty list


for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            DCM_FILES.append(os.path.join(dirName,filename))


RefDs = dicom.read_file(DCM_FILES[0])
PIXEL_DIMENSIONS = (int(RefDs.Rows), int(RefDs.Columns), len(DCM_FILES)) # Z-Axis is the qty of files.
PIXEL_SPACING = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (PIXEL_DIMENSIONS[0]+1)*PIXEL_SPACING[0], PIXEL_SPACING[0])
y = numpy.arange(0.0, (PIXEL_DIMENSIONS[1]+1)*PIXEL_SPACING[1], PIXEL_SPACING[1])
z = numpy.arange(0.0, (PIXEL_DIMENSIONS[2]+1)*PIXEL_SPACING[2], PIXEL_SPACING[2])



# The array is sized based on 'PIXEL_DIMENSIONS'
ArrayDicom = numpy.zeros(PIXEL_DIMENSIONS, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
for filenameDCM in DCM_FILES:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, DCM_FILES.index(filenameDCM)] = ds.pixel_array