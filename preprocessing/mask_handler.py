import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import pydicom_seg
from patchify import unpatchify

from preprocessing import *
from preprocessing import _cropImagesRoutine
from final_dataset_maker import loadImg
from final_dataset_maker import loadMaskImg

dicomImg = pydicom.dcmread(
        "/Volumes/Extreme SSD/CBIS-Mass-Training/Mass-Training_P_00080_RIGHT_CC_FULL.dcm")

imgArray = dicomImg.pixel_array

mskArray = pydicom.dcmread(
        "/Volumes/Extreme SSD/CBIS-Mass-Training/Mass-Training_P_00080_RIGHT_CC_MASK_1.dcm").pixel_array

plt.imshow(imgArray)
plt.show()

def singleProcessingTest(imgArray, mskArray):
    # Get all hyperparameters.
    left = 0.01
    right = 0.01
    up = 0.04
    down = 0.04
    thresh = 0.1
    maxValue = 1.0
    i1 = np.uint8(23)
    kSize = i1
    operation = "open"
    reverse = True
    topXContours = 1
    # outputFormat = ".png"

    # Preprocess full mammogram images.
    fullMammPreprocessed, toLRFlip, nr, nc, leftCrop, rightCrop, upCrop, downCrop = fullMammoPreprocess(
            logger = None,
            img = imgArray,
            left = left,
            right = right,
            up = up,
            down = down,
            thresh = thresh,
            maxval = maxValue,
            ksize = kSize,
            operation = operation,
            reverse = reverse,
            topXContours = topXContours,
            mode = 'cbis',
            reverseInfo = True
    )

    plt.imshow(fullMammPreprocessed)
    plt.show()

    mask_pre = maskPreprocess(logger = None, mask = mskArray, toLRFlip = toLRFlip)

    i, m, leftCrop1, rightCrop1, upCrop1, downCrop1, x, z, y, q, shape0, shape1 = _cropImagesRoutine(
            fullMammPreprocessed,
            mask_pre, reconstructMode = True)

    fixedDim = 2048

    if fixedDim > imgArray.shape[0] or fixedDim > imgArray.shape[1]:
        fixedDim = fixedDim // 2
        if fixedDim > imgArray.shape[0] or fixedDim > imgArray.shape[1]:
            fixedDim = fixedDim // 2

    candidateRows = fixedDim

    while candidateRows < imgArray.shape[0]:
        candidateRows += fixedDim

    candidateCols = fixedDim

    while candidateCols < imgArray.shape[1]:
        candidateCols += fixedDim

    newIm = np.zeros((candidateRows, candidateCols))
    newM = np.zeros((candidateRows, candidateCols))

    oldRows, oldCols = i.shape[0], i.shape[1]

    newIm[:oldRows, :oldCols] = i
    newM[:oldRows, :oldCols] = m

    plt.imshow(newIm)
    plt.show()

    imPatches, dim, shape = loadImg(logger = None, i = newIm, m = newM, mode = 'net_data',
                                    patchify = True)

    # THIS HAS TO BE REMOVED IF THE MODEL IS CALLED, JUST FOR TESTING PURPOSES

    mPatches, _, _ = loadMaskImg(logger = None, m = newM, mode = 'mode', coords = None,
                                 patchify = True)

    ## HERE THE MODEL IS CALLED !!!
    ## REMEMBER TO EXPAND DIMS!!
    ## for image in imPatches:
    ##      mask = model.predict(image)
    ##      mPatches.append(mask)
    ##

    f = lambda x: cv2.resize(x, (dim, dim))

    imPatches = list(map(f, imPatches))
    mPatches = list(map(f, mPatches))

    imPatchesGrid = np.empty(shape)
    mPatchesGrid = np.empty(shape)

    for row in range(0, imPatchesGrid.shape[0]):
        for col in range(0, imPatchesGrid.shape[1]):
            imPatchesGrid[row][col] = imPatches.pop(0)
            mPatchesGrid[row][col] = mPatches.pop(0)

    m = unpatchify(patches = mPatchesGrid, imsize = newM.shape)
    i = unpatchify(patches = imPatchesGrid, imsize = newIm.shape)

    m = m[:oldRows, :oldCols]
    i = i[:oldRows, :oldCols]

    mask_pre00 = np.zeros((shape0, shape1))

    mask_pre00[upCrop1:downCrop1, leftCrop1:rightCrop1] = m

    mask_pre01 = np.zeros_like(mask_pre)

    mask_pre01[y:q, x:z] = mask_pre00

    mask_pre = mask_pre01[:nr, :nc]

    mask_pre = makeLRFlip(None, mask_pre)

    m = np.zeros_like(imgArray)

    m[upCrop:downCrop, leftCrop:rightCrop] = mask_pre

    m = (m == 255).astype(np.uint8) * 255

    return m


def singleProcessing(imgArray):
    # Get all hyperparameters.
    left = 0.01
    right = 0.01
    up = 0.04
    down = 0.04
    thresh = 0.1
    maxValue = 1.0
    i1 = np.uint8(23)
    kSize = i1
    operation = "open"
    reverse = True
    topXContours = 1
    # outputFormat = ".png"

    # Preprocess full mammogram images.
    fullMammPreprocessed, toLRFlip, nr, nc, leftCrop, rightCrop, upCrop, downCrop = fullMammoPreprocess(
            logger = None,
            img = imgArray,
            left = left,
            right = right,
            up = up,
            down = down,
            thresh = thresh,
            maxval = maxValue,
            ksize = kSize,
            operation = operation,
            reverse = reverse,
            topXContours = topXContours,
            mode = 'cbis',
            reverseInfo = True
    )

    # mask_pre = maskPreprocess(logger = None, mask = mskArray, toLRFlip = toLRFlip)

    i, leftCrop1, rightCrop1, upCrop1, downCrop1, x, z, y, q, shape0, shape1 = _cropImagesRoutine(
            i = fullMammPreprocessed,
            m = None, reconstructMode = True)

    fixedDim = 2048

    if fixedDim > imgArray.shape[0] or fixedDim > imgArray.shape[1]:
        fixedDim = fixedDim // 2
        if fixedDim > imgArray.shape[0] or fixedDim > imgArray.shape[1]:
            fixedDim = fixedDim // 2

    candidateRows = fixedDim

    while candidateRows < imgArray.shape[0]:
        candidateRows += fixedDim

    candidateCols = fixedDim

    while candidateCols < imgArray.shape[1]:
        candidateCols += fixedDim

    newIm = np.zeros((candidateRows, candidateCols))

    oldRows, oldCols = i.shape[0], i.shape[1]

    newIm[:oldRows, :oldCols] = i

    imPatches, dim, shape = loadImg(logger = None, i = newIm, m = None, mode = 'net_data',
                                    patchify = True)

    mPatches = []

    ## HERE THE MODEL IS CALLED !!!
    ## REMEMBER TO EXPAND DIMS!!
    ## for image in imPatches:
    ##      mask = model.predict(image)
    ##      mPatches.append(mask)
    ##

    f = lambda x: cv2.resize(x, (dim, dim))
    imPatches = list(map(f, imPatches))
    mPatches = list(map(f, mPatches))

    imPatchesGrid = np.empty(shape)
    mPatchesGrid = np.empty(shape)

    for row in range(0, imPatchesGrid.shape[0]):
        for col in range(0, imPatchesGrid.shape[1]):
            imPatchesGrid[row][col] = imPatches.pop(0)
            mPatchesGrid[row][col] = mPatches.pop(0)

    m = unpatchify(patches = mPatchesGrid, imsize = newIm.shape)
    i = unpatchify(patches = imPatchesGrid, imsize = newIm.shape)

    m = m[:oldRows, :oldCols]

    mask_pre00 = np.zeros((shape0, shape1))

    mask_pre00[upCrop1:downCrop1, leftCrop1:rightCrop1] = m

    mask_pre01 = np.zeros_like(fullMammPreprocessed) # TODO CHECK THIS! CHECK FULMAMMPREPROCESS SIZE IS 2D

    mask_pre01[y:q, x:z] = mask_pre00

    mask_pre = mask_pre01[:nr, :nc]

    mask_pre = makeLRFlip(None, mask_pre)

    m = np.zeros_like(imgArray)

    m[upCrop:downCrop, leftCrop:rightCrop] = mask_pre

    m = (m == 255).astype(np.uint8) * 255

    return m

def overlayMask(img, mask):
    # Stack to create RGB version of grayscale images.
    fullImageRGB = np.stack([img, img, img], axis = -1)

    if fullImageRGB.max() > 255 or fullImageRGB.min() < 0:
        fullImageRGB = fullImageRGB / (fullImageRGB.max() / 255.)

    fullImageRGB = fullImageRGB.astype(np.uint16)
    plt.imshow(fullImageRGB)
    plt.show()

    # Green true mask. Note OpenCV uses BGR.
    trueMasksRGB = np.stack([np.zeros_like(mask), mask, np.zeros_like(mask)], axis = -1)

    if trueMasksRGB.max() > 255 or trueMasksRGB.min() < 0:
        trueMasksRGB = trueMasksRGB / (trueMasksRGB.max() / 255.)

    trueMasksRGB = trueMasksRGB.astype(np.uint16)
    # ------------

    # Then overlay full images and masks.
    segmentedImage = cv2.addWeighted(src1 = fullImageRGB, alpha = 1, src2 = trueMasksRGB, beta = 0.5, gamma = 0)

    return segmentedImage, trueMasksRGB


mask = singleProcessingTest(imgArray, mskArray)
m, t = overlayMask(imgArray, mskArray)
m = m[:,:,1]
m = m.astype(np.uint16)
plt.imshow(m, cmap='gray')
plt.show()

"""# 


dicomImg.PixelData = m.tobytes()
dicomImg.LargestImagePixelValue = 322

print(dicomImg)

plt.imshow(dicomImg.pixel_array)
plt.show()

dicomImg.save_as("img.dcm")"""

import SimpleITK as sitk
import sys
import os

"""reader = sitk.ImageFileReader()
dicom_name = "/Volumes/Extreme SSD/CBIS-Mass-Training/Mass-Training_P_00080_RIGHT_CC_FULL.dcm"
reader.SetFileName(dicom_name)

image = reader.Execute()

size = image.GetSize()
print("Image size:", size[0], size[1], size[2])

mskArray = np.expand_dims(mskArray, 0)
print(mskArray.shape)

segmentation = sitk.GetImageFromArray(mskArray)
segmentation.CopyInformation(image)


template = pydicom_seg.template.from_dcmqi_metainfo('/Users/pablo/Desktop/nl2-project/metainfo.json')

writer = pydicom_seg.MultiClassWriter(
    template=template,
    inplane_cropping=True,  # Crop image slices to the minimum bounding box on
                            # x and y axes
    skip_empty_slices=True,  # Don't encode slices with only zeros
    skip_missing_segment=False,  # If a segment definition is missing in the
                                 # template, then raise an error instead of
                                 # skipping it.
)

dcm = writer.write(segmentation, [dicom_name])

dcm.save_as('segmentation.dcm')
dicomImg.save_as("testttt.dcm")


dicomImg = pydicom.dcmread("/Users/pablo/Desktop/nl2-project/preprocessing/segmentation.dcm")

print(dicomImg)

arr = dicomImg.pixel_array

plt.imshow(arr)
plt.show()"""