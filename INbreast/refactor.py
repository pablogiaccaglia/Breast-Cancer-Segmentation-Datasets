import plistlib

import pandas as pd
import numpy as np
from skimage.draw import polygon
from operator import itemgetter


def loadInbreastMask(mask_path, imshape = (4084, 3328), filter = False):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset

    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]

    return: numpy array where positions in the roi are assigned a value of 1.

    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x

    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt = plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]

            if filter:
                if len(points) < 18:
                    continue

            if len(points) <= 2:
                for point in points:
                    mask[int(point[0]), int(point[1])] = 1
            else:
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = polygon(x, y, shape = imshape)
                mask[poly_x, poly_y] = 1
    return mask


# these acquisitions are removed because without a mass (so no mask!)
def removeBenignAcquisitions(dicomFilenames, csvFilePath, xmlFiles):
    csv = pd.read_csv(csvFilePath, sep = ';')

    column1 = pd.to_numeric(csv['File Name'])
    column2 = csv['Bi-Rads']

    columns = []

    for i in range(len(column1)):
        columns.append((column1[i], column2[i]))

    columns = sorted(columns, key = itemgetter(0))

    dicomFilenames.sort()
    xmlFiles.sort()
    print(len(dicomFilenames))
    print(len(xmlFiles))

    reducedFileNames = []
    reducedXMLFileNames = []
    for i in range(len(xmlFiles)):
        if columns[i][1] != '1':
            reducedFileNames.append(dicomFilenames[i])
            reducedXMLFileNames.append(xmlFiles[i])  # this is actually useless...

    print(len(reducedFileNames))
    print(len(reducedXMLFileNames))

    return reducedFileNames, reducedXMLFileNames

# 26274
