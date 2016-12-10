import dicom
import os
import numpy
from matplotlib import pyplot, cm






BASE_LOCATION = 'D:/Jon\'s Projects/UNMC/Year 3/INDIVIDUAL-DISSERTATION-FYP/Sample-CTs'
PATIENT_FOLDER = '/55144/'

LOCATION = BASE_LOCATION + PATIENT_FOLDER
print(LOCATION)


def addDCM(location):
    for filename in os.listdir(location):
        os.rename(location + filename.title(), location + filename.title() + ".dcm")


def removeAccidentalDCM(location):
    for filename in os.listdir(location):
        os.rename(location + filename.title(), (location + filename.title())[0:-4])
