import os
from distutils.dir_util import copy_tree
from typing import Union

import cv2
import numpy as np
import pydicom
from patchify import patchify
from skimage.exposure import equalize_adapthist

from CBIS.refactorCBIS import updateCSV
from CBIS import handle_multi_tumor
from INbreast.refactorINbreast import removeBenignAcquisitions, loadInbreastMask

from augmentations import pad, padInverted


def cropImageAroundBreast(image,
                          mask = None,
                          doAdditionalCrop = False,
                          reconstructMode = False):
    # Get all hyperparameters.
    thresh = 0.1
    maxValue = 1.0
    i1 = np.uint8(23)
    kSize = i1
    operation = "open"
    reverse = True
    topXContours = 1

    # Step 3: Remove artefacts.
    # i = image[:, :, 0]
    binarizedImage = globalBinarize(logger = None, img = image, thresh = thresh, maxval = maxValue)

    editedMask = editMask(
            logger = None, mask = binarizedImage, ksize = (kSize, kSize), operation = operation
    )
    _, xLargestMask = selectXLargestBlobs(logger = None, mask = editedMask, topXContours = topXContours,
                                          reverse = reverse)
    # cv2.imwrite(
    # "../data/preprocessed/Mass/testing/xLargest_mask.png", xLargestMask
    # )
    # maskedImage = applyMask(logger = logger, img = croppedImage, mask = xLargestMask)
    # cv2.imwrite("../data/preprocessed/Mass/testing/maskedImage.png", maskedImage)
    x, y, z, q = cv2.boundingRect(xLargestMask)

    image = image[y:q, x:z]

    if np.any(mask):
        mask = mask[y:q, x:z]

    if np.any(mask):
        if doAdditionalCrop:
            if reconstructMode:
                image, mask, vertical, mat1, mat2 = cropImage(imag = image, mas = mask, doPad = False,
                                                              reconstructMode = reconstructMode)
                return image, mask, x, vertical, mat1, mat2 + mat1 + y

            else:
                image, mask, _ = cropImage(imag = image, mas = mask, doPad = False)

        return image, mask

    else:
        if doAdditionalCrop:
            if reconstructMode:
                image, _, vertical, mat1, mat2 = cropImage(imag = image, mas = None, doPad = False,
                                                           reconstructMode = reconstructMode)
                return image, _, x, vertical, mat1, mat2 + mat1 + y

            else:
                image, _ = cropImage(imag = image, mas = None, doPad = False)

        return image, mask


def getMaskPatch(img, msk, getBothPatches = False):
    def return_box(x1, x2, y1, y2, multiplier = 1):
        xmid = (x1 + x2) // 2
        ymid = (y1 + y2) // 2
        return xmid + multiplier * (x1 - xmid), \
               ymid + multiplier * (y1 - ymid), \
               xmid + multiplier * (x2 - xmid), \
               ymid + multiplier * (y2 - ymid)

    x, y, w, h = cv2.boundingRect(msk)
    w = max(w, h)
    h = w

    multiplier = 2
    if msk.max() == 255:
        msk = msk / 255
    if msk.sum() < 10000:
        multiplier = 4

    multiplier = 4  # TODO check if it is right to have different multipliers

    x, y, z, q = return_box(x, x + w, y, y + h, multiplier)
    # ima = cv2.rectangle(img, (x, y), (z, q), (0, 255, 0), 2)

    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if z < 0:
        z = 0
    if q < 0:
        q = 0

    if q > img.shape[0]:
        q = img.shape[0]
    if z > img.shape[1]:
        z = img.shape[1]

    croppedImage = img[y:q, x:z]

    if getBothPatches:
        croppedMask = msk[y:q, x:z]
        return croppedImage, croppedMask, [y, q, x, z]

    else:
        return croppedImage, [y, q, x, z]


def getPatches(img, width, height, channels, step):
    if channels is not None:
        patches_img = patchify(img, (width, height, channels),
                               step = step)  # example = Step=256 for 256 patches means no overlap
    else:
        patches_img = patchify(img, (width, height),
                               step = step)  # example = Step=256 for 256 patches means no overlap

    p = []

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            p.append(patches_img[i][j])

    return p, patches_img.shape


def cropImage(imag,
              mas = None,
              doPad = True,
              reconstructMode = False):
    rows = imag.shape[0]
    columns = imag.shape[1]
    mat = None
    newMsk = None
    for i in range(columns - 1, 0, -1):
        value = np.sum(imag[:, i])
        if value > 6000:
            mat = i
            break

    vertical = mat

    newIm = imag[:, :vertical]

    if np.any(mas):
        newMsk = mas[:, :vertical]

    mat1 = None
    for i in range(0, rows):
        value = np.sum(newIm[i, :])
        if value > 20000:
            mat1 = i
            break

    newIm = newIm[mat1:, ]

    if np.any(mas):
        newMsk = newMsk[mat1:, ]

    rows = newIm.shape[0]
    mat2 = None
    for i in range(rows - 1, 0, -1):
        value = np.sum(newIm[i, :])
        if value > 20000:
            mat2 = i
            break

    newIm = newIm[:mat2, ]

    if np.any(mas):
        newMsk = newMsk[:mat2, ]

    newIm = newIm

    if doPad:
        newIm = pad(logger = None, img = newIm, dtype = np.uint8)
        if np.any(mas):
            newMsk = pad(logger = None, img = newMsk, dtype = np.uint8)

    if reconstructMode:
        return newIm, newMsk, vertical, mat1, mat2
    else:
        return newIm, newMsk, vertical


def routineZoomImages(image, mask):
    # image = image[:,:,0]
    # mask = mask[:,:,0]
    image, mask, vertical = cropImage(image, mask)

    x, y, w, h = cv2.boundingRect(mask[:, :, 0])
    # img = cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

    ## get the center and the radius
    cx = x + w // 2
    cy = y + h // 2
    cr = max(w, h) // 2

    ## set offset, repeat enlarger ROI
    dr = 10
    r = cr + 15 * dr
    # cv2.rectangle(im, (cx-r, cy-r), (cx+r, cy+r), (0,255,0), 1)
    toRemove = ((cy - r) // 4)

    yRem0 = cy - r - toRemove
    if yRem0 < 0:
        # toRemove = im.shape[0] - r - cy
        yRem0 = 0

    yRem1 = cy + r + toRemove
    if yRem1 > image.shape[0]:
        # toRemove = im.shape[0] - r - cy
        yRem1 = image.shape[0]

    xRem1 = cx + r + toRemove
    if xRem1 > image.shape[1]:
        # toRemove = im.shape[0] - r - cx
        xRem1 = image.shape[1]

    xRem0 = cx - r - toRemove
    if xRem0 < 0:
        # toRemove = cx-r
        xRem0 = 0

    croppedImage = image[yRem0:yRem1, xRem0:vertical]
    croppedMask = mask[yRem0:yRem1, xRem0:vertical]

    croppedImage = padInverted(img = croppedImage)
    croppedMask = padInverted(img = croppedMask)
    # croppedImage = CLAHE(image = croppedImage)

    return croppedImage, croppedMask


def cropBorders(logger, img,
                left: Union[float, int] = 0.01,
                right: Union[float, int] = 0.01,
                up: Union[float, int] = 0.04,
                down: Union[float, int] = 0.04,
                reverseInfo: bool = False):
    """
    This function crops a specified percentage of border from
    each side of the given image. Default is 1% from the topDirectory,
    left and right sides and 4% from the bottom side.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.
    Returns
    -------
    croppedImg: {numpy.ndarray}
        The cropped image.
    """

    croppedImg = img

    try:
        nrows, ncols = img.shape

        # Get the start and end rows and columns
        leftCrop = int(ncols * left)
        rightCrop = int(ncols * (1 - right))
        upCrop = int(nrows * up)
        downCrop = int(nrows * (1 - down))

        croppedImg = img[upCrop:downCrop, leftCrop:rightCrop]

    except Exception as e:
        # logger.error(f'Unable to cropBorders!\n{e}')
        print(f"Unable to get cropBorders!\n{e}")

    if reverseInfo:
        return croppedImg, leftCrop, rightCrop, upCrop, downCrop
    else:
        return croppedImg


def minMaxNormalize(logger, img: np.ndarray) -> np.ndarray:
    """
    This function does min-max normalisation [0,1] on
    the given image.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to normalize.
    Returns
    -------
    normalizedImg: {numpy.ndarray}
        The min-max normalized image.
    """

    normalizedImg = img

    try:
        normalizedImg = (img - img.min()) / (img.max() - img.min())

    except Exception as e:
        # logger.error(f'Unable to minMaxNormalise!\n{e}')
        print((f"Unable to get minMaxNormalise!\n{e}"))

    return normalizedImg


def setBlackRegion(logger, img: np.ndarray, start: int, end: int) -> np.ndarray:
    for i in range(img.shape[0]):
        for j in range(int(start), end):
            img[i][j] = 0

    return img


def globalBinarize(logger, img: np.ndarray, thresh, maxval: float) -> np.ndarray:
    """
    This function takes in a numpy array image and
    returns a corresponding mask that is a global
    binarization on it based on a given threshold
    and maxval. Any elements in the array that is
    greater than or equals to the given threshold
    will be assigned maxval, else zero.
    Parameters
    ----------

    img : {numpy.ndarray}
        The image to perform binarization on.
    thresh : {int or float}
        The global threshold for binarization.
    maxval : {np.uint8}
        The value assigned to an element that is greater than or equals to `thresh`.
    Returns
    -------
    binarizedImage : {numpy.ndarray, dtype=np.uint8}
        A binarized image of {0, 1}.
    """

    binarizedImage = img

    try:
        binarizedImage = np.zeros(img.shape, np.uint8)
        binarizedImage[img >= thresh] = maxval

    except Exception as e:
        # logger.error(f'Unable to globalBinarise!\n{e}')
        print(f"Unable to globalBinarise!\n{e}")

    return binarizedImage


def editMask(logger, mask: np.ndarray, ksize: tuple, operation: str = "open") -> np.ndarray:
    """
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.
    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.
    ksize : {tuple}
        Size of the structuring element.
    operation : {str}
        Either "open" or "close", each representing open and close
        morphological operations respectively.
    Returns
    -------
    editedMask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    """

    editedMask = mask

    try:
        kernel = cv2.getStructuringElement(shape = cv2.MORPH_RECT, ksize = ksize)

        if operation == "open":
            editedMask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            editedMask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Then dilate
        editedMask = cv2.morphologyEx(editedMask, cv2.MORPH_DILATE, kernel)

    except Exception as e:
        # logger.error(f'Unable to editMask!\n{e}')
        print(f"Unable to get editMask!\n{e}")

    return editedMask


def sortContoursByArea(logger = None, contours: list = None, reverse = True) -> tuple:
    """
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.
    Parameters
    ----------
    contours : {list}
        The list of contours to sort.

    reverse : {bool}
    Returns
    -------
    sortedContours : {list}
        The list of contours sorted by contour area in descending
        order.
    boundingBoxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sortedContours`.
    """

    if contours is None:
        contours = []
    sortedContours = contours
    boundingBoxes = [cv2.boundingRect(c) for c in sortedContours]

    try:
        # Sort contours based on contour area.
        sortedContours = sorted(contours, key = cv2.contourArea, reverse = reverse)

        # Construct the list of corresponding bounding boxes.
        boundingBoxes = [cv2.boundingRect(c) for c in sortedContours]

    except Exception as e:
        # logger.error(f'Unable to sortContourByArea!\n{e}')
        print(f"Unable to get sortContourByArea!\n{e}")

    return sortedContours, boundingBoxes


def selectXLargestBlobs(logger, mask: np.ndarray, topXContours: int = None, reverse: bool = True) -> tuple:
    """
    This function finds contours in the given image and
    keeps only the topDirectory X largest ones.
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the topDirectory X largest blobs.
    topXContours : {int}
        The topDirectory X contours to keep based on contour area
        ranked in decesnding order.

    reverse: {bool} ...

    Returns
    -------
    numOfContours : {int}
        The number of contours found in the given `mask`.
    XLargestBlobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the topDirectory X largest contours in white.

    """

    numOfContours = 0

    try:
        # Find all contours from binarised image.
        # Note: parts of the image that you want to get should be white.
        contours, hierarchy = cv2.findContours(
                image = mask, mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE
        )

        numOfContours = len(contours)

        # Only get largest blob if there is at least 1 contour.
        if numOfContours > 0:

            # Make sure that the number of contours to keep is at most equal
            # to the number of contours present in the mask.
            if numOfContours < topXContours or topXContours is None:
                topXContours = numOfContours

            # Sort contours based on contour area.
            sortedContours, boundingBoxes = sortContoursByArea(logger = None,
                                                               contours = contours, reverse = reverse
                                                               )

            # Get the topDirectory X largest contours.
            XLargestContours = sortedContours[0:topXContours]

            # Create black canvas to draw contours on.
            imageToDrawOn = np.zeros(mask.shape, np.uint8)

            # Draw contours in XLargestContours.
            xLargestBlobs = cv2.drawContours(
                    image = imageToDrawOn,  # Draw the contours on `imageToDrawOn`.
                    contours = XLargestContours,  # List of contours to draw.
                    contourIdx = -1,  # Draw all contours in `contours`.
                    color = 1,  # Draw the contours in white.
                    thickness = -1,  # Thickness of the contour lines.
            )

    except Exception as e:
        # logger.error(f'Unable to xLargestBlobs!\n{e}')
        print(f"Unable to get xLargestBlobs!\n{e}")

    return numOfContours, xLargestBlobs


def applyMask(logger, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    maskedImage = img.copy()
    maskedImage[mask == 0] = 0

    return maskedImage


def checkLRFlip(logger, mask: np.ndarray) -> bool:
    """
    This function checks whether or not an image needs to be
    flipped horizontally (i.e. left-right flip). The correct
    orientation is the breast being on the left (i.e. facing
    right).
    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the image to flip.
    Returns
    -------
    toFlip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    """

    toFlip = False

    try:
        # Get number of rows and columns in the image.
        nRows, nCols = mask.shape
        xCenter = nCols // 2

        # yCenter = nRows // 2

        # Sum down each column.
        colSum = mask.sum(axis = 0)

        # Sum across each row.
        # rowSum = mask.sum(axis=1)

        leftSum = sum(colSum[0:xCenter])
        rightSum = sum(colSum[xCenter:-1])

        if leftSum < rightSum:
            toFlip = True
        else:
            toFlip = False

    except Exception as e:
        # logger.error(f'Unable to checkLRFlip!\n{e}')
        print(f"Unable to get checkLRFlip!\n{e}")

    return toFlip


def makeLRFlip(logger, img: np.ndarray) -> np.ndarray:
    """
    This function flips a given image horizontally (i.e. left-right).
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to flip.
    Returns
    -------
    flippedImg : {numpy.ndarray}
        The flipped image.
    """

    flippedImage = img

    try:
        flippedImage = np.fliplr(img)
    except Exception as e:
        # logger.error(f'Unable to makeLRFlip!\n{e}')
        print(f"Unable to get makeLRFlip!\n{e}")

    return flippedImage


def fullMammoPreprocess(
        logger,
        img: np.ndarray,
        left: Union[float, int],
        right: Union[float, int],
        down: Union[float, int],
        up: Union[float, int],
        thresh,
        maxval: float,
        ksize: np.uint8,
        operation: str,
        reverse: bool,
        topXContours: int,
        mode: str,
        mammMask: np.ndarray = None,
        reverseInfo: bool = False
) -> tuple:
    """
    This function chains and executes all the preprocessing
    steps for a full mammogram, in the following order:
    Step 1 - Initial crop
    Step 2 - Min-max normalise
    Step 3 - Remove artefacts
    Step 4 - Horizontal flip (if needed)
    Step 5 - CLAHE enchancement
    Step 6 - Pad
    Step 7 - Downsample (?)
    Step 8 - Min-max normalise
    Parameters
    ----------
    img : {numpy.ndarray}
        The full mammogram image to preprocess.
    Returns
    -------
    imgPreprocessed : {numpy.ndarray}
        The preprocessed full mammogram image.
    toLRFlip : {boolean}
        If True, the corresponding ROI mask needs to be
        flipped horizontally, otherwise no need to flip.
    """

    imgPreprocessed, toLRFlip = img, True

    try:

        if mode == 'csaw':
            binarizedMask = globalBinarize(logger = logger, img = mammMask, thresh = thresh, maxval = maxval)
            img = applyMask(logger = logger, img = img, mask = binarizedMask)

        # Step 1: Initial crop.
        if reverseInfo:
            croppedImage, leftCrop, rightCrop, upCrop, downCrop = cropBorders(logger = logger,
                                                                              img = img,
                                                                              left = left,
                                                                              right = right,
                                                                              down = down,
                                                                              up = up,
                                                                              reverseInfo = reverseInfo)
        else:
            croppedImage = cropBorders(logger = logger,
                                       img = img,
                                       left = left,
                                       right = right,
                                       down = down,
                                       up = up,
                                       reverseInfo = reverseInfo)

        # cv2.imwrite("../data/preprocessed/Mass/testing/cropped.png", croppedImage)

        # Step 2: Min-max normalize.
        normalizedImage = minMaxNormalize(logger = logger, img = croppedImage)
        # cv2.imwrite("../data/preprocessed/Mass/testing/normed.png", normalizedImage)

        if mode == 'cbis':
            # Step 3: Remove artefacts.
            binarizedImage = globalBinarize(logger = logger, img = normalizedImage, thresh = thresh, maxval = maxval)
            editedMask = editMask(
                    logger = logger, mask = binarizedImage, ksize = (ksize, ksize), operation = operation
            )
            _, xLargestMask = selectXLargestBlobs(logger = logger, mask = editedMask, topXContours = topXContours,
                                                  reverse = reverse)
            # cv2.imwrite(
            # "../data/preprocessed/Mass/testing/xLargest_mask.png", xLargestMask
            # )

            maskedImage = applyMask(logger = logger, img = croppedImage, mask = xLargestMask)
            # cv2.imwrite("../data/preprocessed/Mass/testing/maskedImage.png", maskedImage)


        else:
            xLargestMask = croppedImage
            maskedImage = croppedImage

        # Step 4: Horizontal flip.
        toLRFlip = checkLRFlip(logger = logger, mask = xLargestMask)

        if toLRFlip:
            flippedImage = makeLRFlip(logger = logger, img = maskedImage)
        elif not toLRFlip:
            flippedImage = maskedImage
        # cv2.imwrite("../data/preprocessed/Mass/testing/flippedImage.png", flippedImage)

        # Step 5: CLAHE enhancement.
        if mode == 'cbis':  # CLAHE not needed for CSAW images!
            claheImage = equalize_adapthist(flippedImage)
            # claheImage = flippedImage

        else:
            claheImage = flippedImage

        # cv2.imwrite("../data/preprocessed/Mass/testing/claheImage.png", claheImage)

        # Step 6: pad.
        paddedImage = pad(logger = logger, img = claheImage)

        paddedImage = cv2.normalize(
                paddedImage,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )
        # cv2.imwrite("../data/preprocessed/Mass/testing/paddedImage.png", paddedImage)

        # Step 7: Downsample.
        # Not done yet.

        # Step 8: Min-max normalise.
        # imgPreprocessed = minMaxNormalize(logger = logger, img = paddedImage)
        # cv2.imwrite("../data/preprocessed/Mass/testing/imgPreprocessed.png", imgPreprocessed)

    except Exception as e:
        # logger.error(f'Unable to fullMammPreprocess!\n{e}')
        print(f"Unable to fullMammPreprocess!\n{e}")

    if reverseInfo:
        return paddedImage, toLRFlip, croppedImage.shape[0], croppedImage.shape[
            1], leftCrop, rightCrop, upCrop, downCrop

    else:
        return paddedImage, toLRFlip


def maskPreprocess(logger, mask: np.ndarray, toLRFlip: bool) -> np.ndarray:
    """
    This function chains and executes all the preprocessing
    steps necessary for a ROI mask image.
    Step 1 - Initial crop
    Step 2 - Horizontal flip (if needed)
    Step 3 - Pad
    Step 4 - Downsample (?)
    Parameters
    ----------
    mask : {numpy.ndarray}
        The ROI mask image to preprocess.
    toLRFlip : {boolean}
        If True, the ROI mask needs to be
        flipped horizontally, otherwise no need to flip.
    Returns
    -------
    maskPreprocessed : {numpy.ndarray}
        The preprocessed ROI mask image.
    """

    # Step 1: Initial crop.
    mask = cropBorders(logger, img = mask)

    # Step 2: Horizontal flip.
    if toLRFlip:
        mask = makeLRFlip(logger, img = mask)

    # Step 3: Pad.
    maskPreprocessed = pad(logger = logger, img = mask)

    # Step 4: Downsample.

    return maskPreprocessed


def sumMasks(logger, mask_list):
    """
    This function sums a list of given masks.
    Parameters
    ----------
    mask_list : {list of numpy.ndarray}
        A list of masks (numpy.ndarray) that needs to be summed.
    Returns
    -------
    summed_mask_bw: {numpy.ndarray}
        The summed mask, ranging from [0, 1].
    """

    try:

        summed_mask = np.zeros(mask_list[0].shape)

        for arr in mask_list:
            summed_mask = np.add(summed_mask, arr)

        # Binarise (there might be some overlap, resulting in pixels with
        # values of 510, 765, etc...)
        _, summed_mask_bw = cv2.threshold(
                src = summed_mask, thresh = 1, maxval = 255, type = cv2.THRESH_BINARY
        )

    except Exception as e:
        # logger.error(f'Unable to findMultiTumour!\n{e}')
        print((f"Unable to get findMultiTumour!\n{e}"))

    return


def CBISPreprocessing(logger, imagesPath: str, outputImagesPath: str, outputMasksPath: str, suffix: str):
    """main function for imagePreprocessing module.
    This function takes a i of the raw image folder,
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images (in the specified file extension)
    in the output paths specified. FULL and MASK images are
    saved in their specified separate folders.

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    """

    # Get individual .dcm paths.
    dcmPaths = []

    for currentDirectory, dirs, files in os.walk(imagesPath):
        files.sort()
        for f in files:
            if f.endswith(".dcm"):
                dcmPaths.append(os.path.join(currentDirectory, f))

    # Get paths of full mammograms and ROI masks.
    fullMammPaths = [f for f in dcmPaths if ("FULL" in f and '___a.dcm' not in f)]
    masksPaths = [f for f in dcmPaths if ("MASK" in f and '___a.dcm' not in f)]

    for fullMammPath in fullMammPaths:
        # Read full mammogram .dcm file.
        fullMammPath = fullMammPath.replace('._', '')
        ds = pydicom.dcmread(fullMammPath)

        # Get relevant metadata from .dcm file.
        patientID = ds.PatientID

        # Calc-Test masks do not have the "Calc-Test_" suffix
        # when it was originally downloaded (masks from Calc-Train,
        # Mass-Test and Mass-Train all have their corresponding suffices).
        patientID = patientID.replace(".dcm", "")
        patientID = patientID.replace("Calc-Test_", "")
        patientID = patientID.replace("Mass-Test_", "")

        patientID = patientID.replace("Mass-Training_", "")
        patientID = patientID.replace("Calc_Training_", "")

        # sanity check for images, some calc images dcm files are corrupted,
        # so we ignore these entries
        try:
            fullMamm = ds.pixel_array
        except:
            continue

        # sanity check for masks, some calc mask dcm files are corrupted,
        # so we ignore these entries

        # Get the i of corresponding ROI mask(s) .dcm file(s).
        maskImagePath = [mp for mp in masksPaths if patientID in mp]
        maskArrays = []
        maskImagesPathUpdated = []

        for mp in maskImagePath:
            try:
                # Read mask(s) .dcm file(s).
                mp = mp.replace('._', '')
                maskDcm = pydicom.dcmread(mp)
                maskArray = maskDcm.pixel_array
                maskArrays.append(maskArray)
                maskImagesPathUpdated.append(mp)
            except:
                pass

        maskImagePath = maskImagesPathUpdated

        if len(maskArrays) == 0:
            continue
        # =========================
        # Preprocess Full Mammogram
        # =========================

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
        outputFormat = ".png"

        # Preprocess full mammogram images.
        fullMammPreprocessed, toLRFlip = fullMammoPreprocess(
                logger = None,
                img = fullMamm,
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
                mode = 'cbis'
        )

        # Need to normalise to [0, 255] before saving as .png.
        """fullMammPreprocessedNormalized = cv2.normalize(
                fullMammPreprocessed,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )"""

        # Save preprocessed full mammogram image.
        savedFilename = (
                os.path.basename(fullMammPath).replace(".dcm", "")
                + "___PRE"
                + outputFormat
        )

        if suffix not in savedFilename:
            savedFilename = suffix + savedFilename

        savedFilePath = os.path.join(outputImagesPath, savedFilename)

        shape = 0

        for maskArray, mp in zip(maskArrays, maskImagePath):
            # Preprocess.
            mask_pre = maskPreprocess(logger = None, mask = maskArray, toLRFlip = toLRFlip)

            # Save preprocessed mask.
            savedFilename = (
                    os.path.basename(mp).replace(".dcm", "") + "___PRE" + outputFormat
            )

            if suffix not in savedFilename:
                savedFilename = suffix + savedFilename

            savedFilePathMask = os.path.join(outputMasksPath, savedFilename)

            s1 = min(fullMammPreprocessed.shape[0], mask_pre.shape[0])
            s2 = min(mask_pre.shape[1], fullMammPreprocessed.shape[1])

            shape = min(s1, s2)

            mask_pre = cv2.resize(mask_pre, (shape, shape))

            cv2.imwrite(savedFilePathMask, mask_pre)

        #  print(f"DONE MASK: {mp}")

        fullMammPreprocessed = cv2.resize(fullMammPreprocessed, (shape, shape))
        cv2.imwrite(savedFilePath, fullMammPreprocessed)
        # print(f"DONE FULL: {fullMammPath}")


def getListOfFiles(dirPath: str, extension: str) -> list:
    listOfPaths = []

    for currentDirectory, dirs, files in os.walk(dirPath):
        files.sort()
        for f in files:
            if f.endswith(extension):
                listOfPaths.append(os.path.join(currentDirectory, f))

    return listOfPaths


def CSAWPreprocessing(logger, imagesPath: str, masksPath: str, mammGlandMasksPath: str, outputImagesPath: str,
                      outputMasksPath: str):
    imagesPaths = getListOfFiles(dirPath = imagesPath, extension = ".png")
    masksPaths = getListOfFiles(dirPath = masksPath, extension = ".png")
    mammGlandsPaths = getListOfFiles(dirPath = mammGlandMasksPath, extension = ".png")

    count = 0

    masksPaths.sort()
    imagesPaths.sort()
    mammGlandsPaths.sort()

    for fullMammPath in imagesPaths:
        # Read full mammogram .png file.
        fullMamm = cv2.imread(fullMammPath, cv2.IMREAD_GRAYSCALE)

        mammID = fullMammPath.split('-')[-1]

        mammGlandMaskPath = [p for p in mammGlandsPaths if p.split('-')[-1] == mammID][0]

        mammGlandMask = cv2.imread(mammGlandMaskPath, cv2.IMREAD_GRAYSCALE)

        # =========================
        # Preprocess Full Mammogram
        # =========================

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
        outputFormat = ".png"

        # Preprocess full mammogram images.
        fullMammPreprocessed, toLRFlip = fullMammoPreprocess(
                logger = None,
                img = fullMamm,
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
                mode = 'csaw',
                mammMask = mammGlandMask
        )

        # Need to normalise to [0, 255] before saving as .png.
        """fullMammPreprocessedNormalized = cv2.normalize(
                fullMammPreprocessed,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )"""

        # Save preprocessed full mammogram image.
        savedFilename = (
                os.path.basename(fullMammPath)
                + "___PRE"
                + outputFormat
        )
        savedFilePath = os.path.join(outputImagesPath, savedFilename)

        cv2.imwrite(savedFilePath, fullMammPreprocessed)
        # print(f"DONE FULL: {fullMammPath}")

        # ================================
        # Preprocess Corresponding Mask(s)
        # ================================

        # Get the i of corresponding ROI mask(s) .dcm file(s).

        masksPathsList = [i for i in masksPaths if i.split('-')[-1] == mammID]

        for mp in masksPathsList:
            maskArray = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

            # Preprocess.
            mask_pre = maskPreprocess(logger = None, mask = maskArray, toLRFlip = toLRFlip)

            # Save preprocessed mask.
            savedFilename = (
                    os.path.basename(mp).replace(".dcm", "") + "___PRE" + outputFormat
            )
            savedFilePath = os.path.join(outputMasksPath, savedFilename)
            cv2.imwrite(savedFilePath, mask_pre)

        #  print(f"DONE MASK: {mp}")

        count += 1

        # if count == 1:
        #     break

    print(f"Total count = {count}")
    print()
    print("Getting out of imagePreprocessing module.")
    print("-" * 30)

    return


def __CBISParametricRoutine(dcmFolder: str,
                            originalPreprocessedIMGFolderPath: str,
                            originalPreprocessedMSKFolderPath: str,
                            originalCSVPath: str,
                            updatedCSVPath: str,
                            completePreprocessedIMGFolderPath: str,
                            completePreprocessedMSKFolderPath: str,
                            abnormality_col: str,
                            extension: str, suffix: str) -> None:
    CBISPreprocessing(logger = None, imagesPath = dcmFolder,
                      outputImagesPath = originalPreprocessedIMGFolderPath,
                      outputMasksPath = originalPreprocessedMSKFolderPath, suffix = suffix)

    updateCSV(logger = None, mass_csv_path = originalCSVPath,
              mass_png_folder = originalPreprocessedIMGFolderPath,
              masks_folder = originalPreprocessedMSKFolderPath,
              output_csv_path = updatedCSVPath)

    handle_multi_tumor.handleMultiTumor(csv_path = updatedCSVPath,
                                        abnormality_col = abnormality_col,
                                        img_path = originalPreprocessedIMGFolderPath,
                                        masks_path = originalPreprocessedMSKFolderPath,
                                        output_path = originalPreprocessedMSKFolderPath,
                                        extension = extension, suffix = suffix)

    copy_tree(originalPreprocessedIMGFolderPath, completePreprocessedIMGFolderPath)
    copy_tree(originalPreprocessedMSKFolderPath, completePreprocessedMSKFolderPath)


def _cropImagesRoutine(i, m = None, reconstructMode = False, doAdditionalCrop = True):
    if np.any(m):
        s1 = min(i.shape[0], m.shape[0])
        s2 = min(i.shape[1], m.shape[1])

        shape = min(s1, s2)

        m = cv2.resize(m, (shape, shape))
        i = cv2.resize(i, (shape, shape))

    if np.any(m):
        print(m.shape)

    if reconstructMode:
        if doAdditionalCrop:
            i, m, x, z, y, q = cropImageAroundBreast(
                    image = i,
                    mask = m,
                    doAdditionalCrop = doAdditionalCrop,
                    reconstructMode = reconstructMode)
        else:
            i, m = cropImageAroundBreast(
                    image = i,
                    mask = m,
                    doAdditionalCrop = doAdditionalCrop,
                    reconstructMode = reconstructMode)


    else:
        i, m = cropImageAroundBreast(image = i,
                                     mask = m,
                                     doAdditionalCrop = doAdditionalCrop)

    if np.any(m):
        if reconstructMode:
            i1, leftCrop, rightCrop, upCrop, downCrop = cropBorders(logger = None, img = i,
                                                                    reverseInfo = reconstructMode)
            m = cropBorders(logger = None, img = m)
            return i1, m, leftCrop, rightCrop, upCrop, downCrop, x, z, y, q, i.shape[0], i.shape[1]
        else:
            i = cropBorders(logger = None, img = i)
            m = cropBorders(logger = None, img = m)
            return i, m


    else:
        if reconstructMode:
            i1, leftCrop, rightCrop, upCrop, downCrop = cropBorders(logger = None, img = i,
                                                                    reverseInfo = reconstructMode)
            return i1, leftCrop, rightCrop, upCrop, downCrop, x, z, y, q, i.shape[0], i.shape[1]

        else:
            i = cropBorders(logger = None, img = i)
            return i


def CropAndSaveImagesRoutine(MassImagesPath,
                             MassMasksPath,
                             croppedMassImagesPath,
                             croppedMassMasksPath):
    from final_dataset_maker import datasetPaths

    # Get paths to individual images.
    images, masks = datasetPaths(
            logger = None,
            full_img_dir = MassImagesPath,
            mask_img_dir = MassMasksPath,
            extension = ".png",
    )

    for iPath, mPath in zip(images, masks):
        i = cv2.imread(iPath, cv2.cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(mPath, cv2.cv2.IMREAD_GRAYSCALE)

        i, m = _cropImagesRoutine(i = i,
                                  m = m)

        _, maskFileName = os.path.split(mPath)
        maskFileName = 'CROPPED-' + maskFileName

        _, imgFileName = os.path.split(iPath)
        imgFileName = 'CROPPED-' + imgFileName

        cv2.imwrite(os.path.join(croppedMassImagesPath, imgFileName), i)
        cv2.imwrite(os.path.join(croppedMassMasksPath, maskFileName), m)


def CBISFullRoutine():
    dcmMassTrainingFolderPath = "/Volumes/Extreme SSD//CBIS-Mass-Training/"
    dcmMassTestingFolderPath = "/Volumes/Extreme SSD//CBIS-Mass-Testing/"

    dcmCalcTrainingFolderPath = "/Volumes/Extreme SSD//CBIS-Calc-Training/"
    dcmCalcTestingFolderPath = "/Volumes/Extreme SSD//CBIS-Calc-Testing/"

    originalTrainingMassPreprocessedIMGFolderPath = "../CBIS/original-preprocessed/intermediate/CBIS-Original-Mass-Training-Preprocessed-IMG"
    originalTrainingMassPreprocessedMSKFolderPath = "../CBIS/original-preprocessed/intermediate/CBIS-Original-Mass-Training-Preprocessed-MSK"
    originalTrainingMassCSVPath = "../CBIS/csv/mass_case_description_train_set.csv"
    updatedTrainingMassCSVPath = "../CBIS/csv/mass_case_description_train_set_UPDATED2.csv"

    originalTestingMassPreprocessedIMGFolderPath = ".../CBIS/original-preprocessed/intermediate/CBIS-Original-Mass-Testing-Preprocessed-IMG"
    originalTestingMassPreprocessedMSKFolderPath = "../CBIS/original-preprocessed/intermediate/CBIS-Original-Mass-Testing-Preprocessed-MSK"

    originalTestingMassCSVPath = "../CBIS/csv/mass_case_description_test_set.csv"
    updatedTestingMassCSVPath = "../CBIS/csv/mass_case_description_test_set_UPDATED2.csv"

    completePreprocessedMassIMGFolderPath = "../CBIS-Original-Mass-Preprocessed-Complete-IMG"
    completePreprocessedMassMSKFolderPath = "../CBIS-Original-Mass-Preprocessed-Complete-MSK"

    originalTrainingCalcPreprocessedIMGFolderPath = "..CBIS/original-preprocessed/intermediate/CBIS-Original-Calc-Training-Preprocessed-IMG"
    originalTrainingCalcPreprocessedMSKFolderPath = "...CBIS/original-preprocessed/intermediate/CBIS-Original-Calc-Training-Preprocessed-MSK"
    originalTrainingCalcCSVPath = "../CBIS/csv/calc_case_description_train_set.csv"
    updatedTrainingCalcCSVPath = "../CBIS/csv/calc_case_description_train_set_UPDATED.csv"

    originalTestingCalcPreprocessedIMGFolderPath = "...CBIS/original-preprocessed/intermediate/CBIS-Original-Calc-Testing-Preprocessed-IMG"
    originalTestingCalcPreprocessedMSKFolderPath = "..CBIS/original-preprocessed/intermediate/CBIS-Original-Calc-Testing-Preprocessed-MSK"
    originalTestingCalcCSVPath = "../CBIS/csv/calc_case_description_test_set.csv"
    updatedTestingCalcCSVPath = "../CBIS/csv/calc_case_description_test_set_UPDATED.csv"

    completePreprocessedCalcIMGFolderPath = "..CBIS/original-preprocessed/CBIS-Original-Calc-Preprocessed-Complete-IMG"
    completePreprocessedCalcMSKFolderPath = "...CBIS/original-preprocessed/CBIS-Original-Calc-Preprocessed-Complete-MSK"

    abnormality_col = "abnormality_id"
    extension = ".png"

    __CBISParametricRoutine(dcmFolder = dcmMassTrainingFolderPath,
                            originalPreprocessedIMGFolderPath = originalTrainingMassPreprocessedIMGFolderPath,
                            originalPreprocessedMSKFolderPath = originalTrainingMassPreprocessedMSKFolderPath,
                            originalCSVPath = originalTrainingMassCSVPath,
                            updatedCSVPath = updatedTrainingMassCSVPath,
                            completePreprocessedIMGFolderPath = completePreprocessedMassIMGFolderPath,
                            completePreprocessedMSKFolderPath = completePreprocessedMassMSKFolderPath,
                            abnormality_col = abnormality_col,
                            extension = extension,
                            suffix = 'Mass-Training_')

    __CBISParametricRoutine(dcmFolder = dcmMassTestingFolderPath,
                            originalPreprocessedIMGFolderPath = originalTestingMassPreprocessedIMGFolderPath,
                            originalPreprocessedMSKFolderPath = originalTestingMassPreprocessedMSKFolderPath,
                            originalCSVPath = originalTestingMassCSVPath,
                            updatedCSVPath = updatedTestingMassCSVPath,
                            completePreprocessedIMGFolderPath = completePreprocessedMassIMGFolderPath,
                            completePreprocessedMSKFolderPath = completePreprocessedMassMSKFolderPath,
                            abnormality_col = abnormality_col,
                            extension = extension,
                            suffix = 'Mass-Test_')

    """__CBISParametricRoutine(dcmFolder = dcmCalcTrainingFolderPath,
                            originalPreprocessedIMGFolderPath = originalTrainingCalcPreprocessedIMGFolderPath,
                            originalPreprocessedMSKFolderPath = originalTrainingCalcPreprocessedMSKFolderPath,
                            originalCSVPath = originalTrainingCalcCSVPath,
                            updatedCSVPath = updatedTrainingCalcCSVPath,
                            completePreprocessedIMGFolderPath = completePreprocessedCalcIMGFolderPath,
                            completePreprocessedMSKFolderPath = completePreprocessedCalcMSKFolderPath,
                            abnormality_col = abnormality_col,
                            extension = extension,
                            suffix = 'Calc-Training_')

    __CBISParametricRoutine(dcmFolder = dcmCalcTestingFolderPath,
                            originalPreprocessedIMGFolderPath = originalTestingCalcPreprocessedIMGFolderPath,
                            originalPreprocessedMSKFolderPath = originalTestingCalcPreprocessedMSKFolderPath,
                            originalCSVPath = originalTestingCalcCSVPath,
                            updatedCSVPath = updatedTestingCalcCSVPath,
                            completePreprocessedIMGFolderPath = completePreprocessedCalcIMGFolderPath,
                            completePreprocessedMSKFolderPath = completePreprocessedCalcMSKFolderPath,
                            abnormality_col = abnormality_col,
                            extension = extension,
                            suffix = 'Calc-Test_')"""

    CropAndSaveImagesRoutine(MassImagesPath = "../CBIS/CBIS-MASS-SELECTED-IMGS",
                             MassMasksPath = "../CBIS/CBIS-MASS-SELECTED-MASKS",
                             croppedMassImagesPath = "../CBIS/cropped/CBIS-MASS-Cropped-FINAL-IMG",
                             croppedMassMasksPath = "../CBIS/cropped/CBIS-MASS-Cropped-FINAL-MSK")


def CSAWFullRoutine():
    originalCSAWImagesPath = "../CSAW/original/CSAW-Original-IMG"
    originalCSAWMasksPath = "../CSAW/original/CSAW-Original-MSK"
    mammGlandsMasksPath = "../CSAW/original/CSAW-Original-Mammary-Gland"

    originalPreprocessedCSAWImagesPath = "../CSAW/original/CSAW-Original-Preprocessed-IMG"
    originalPreprocessedCSAWMakskPath = "../CSAW/original/CSAW-Original-Preprocessed-MSK"



    CSAWPreprocessing(logger = None,
                      imagesPath = originalCSAWImagesPath,
                      masksPath = originalCSAWMasksPath,
                      outputImagesPath = originalPreprocessedCSAWImagesPath,
                      outputMasksPath = originalPreprocessedCSAWMakskPath,
                      mammGlandMasksPath = mammGlandsMasksPath)

    """CropAndSaveImagesRoutine(MassImagesPath = "../CSAW/CSAW-SELECTED-IMGS",
                             MassMasksPath = "../CSAW/CSAW-SELECTED-MASKS",
                             croppedMassImagesPath = "../CSAW/CSAW-Cropped-FINAL-IMG",
                             croppedMassMasksPath = "../CSAW/CSAW-Cropped-FINAL-MSK")"""


def BCDRPreprocessing(imagesDirPath, masksDirPath, outputImagesPath, outputMasksPath):
    # Get individual .tif paths.
    imagesPaths = []

    for currentDirectory, dirs, files in os.walk(imagesDirPath):
        files.sort()
        for f in files:
            if f.endswith(".tif"):
                imagesPaths.append(os.path.join(currentDirectory, f))

    # Get individual xml paths.
    masksPaths = []

    for currentDirectory, dirs, files in os.walk(masksDirPath):
        files.sort()
        for f in files:
            if f.endswith(".tif"):
                masksPaths.append(os.path.join(currentDirectory, f))

    count = 0
    for fullMammPath in imagesPaths:

        imgID = fullMammPath.split('/')[-1]
        imgID = imgID[:-4]

        mp = [i for i in masksPaths if imgID in i]

        if len(mp) == 0:
            continue

        mp = mp[0]

        maskArray = cv2.imread(filename = mp, flags = cv2.cv2.IMREAD_GRAYSCALE)
        if np.sum(maskArray) < 6000:
            continue

        # Read full mammogram .dcm file.
        fullMamm = cv2.imread(filename = fullMammPath, flags = cv2.cv2.IMREAD_GRAYSCALE)

        # =========================
        # Preprocess Full Mammogram
        # =========================

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
        outputFormat = ".png"

        # Preprocess full mammogram images.
        fullMammPreprocessed, toLRFlip = fullMammoPreprocess(
                logger = None,
                img = fullMamm,
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
                mode = 'cbis'
        )

        # Need to normalise to [0, 255] before saving as .png.
        """fullMammPreprocessedNormalized = cv2.normalize(
                fullMammPreprocessed,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )"""

        # Save preprocessed full mammogram image.
        savedFilename = (
                os.path.basename(fullMammPath).replace(".tif", "")
                + "___PRE"
                + outputFormat
        )
        savedFilePath = os.path.join(outputImagesPath, savedFilename)

        cv2.imwrite(savedFilePath, fullMammPreprocessed)
        # print(f"DONE FULL: {fullMammPath}")

        # ================================
        # Preprocess Corresponding Mask(s)
        # ================================

        # Preprocess.
        mask_pre = maskPreprocess(logger = None, mask = maskArray, toLRFlip = toLRFlip)

        mask_pre = cv2.normalize(
                mask_pre,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )

        # Save preprocessed mask.
        savedFilename = (
                os.path.basename(mp).replace(".tif", "") + "___PRE" + outputFormat
        )
        savedFilePath = os.path.join(outputMasksPath, savedFilename)
        cv2.imwrite(savedFilePath, mask_pre)

        #  print(f"DONE MASK: {mp}")

        count += 1

        # if count == 1:
        #     break

    print(f"Total count = {count}")
    print()
    print("Getting out of imagePreprocessing module.")
    print("-" * 30)

    return


def INBreastFullRoutine():
    originalINBreastImagesPath = "../INbreast/original/AllDICOMs"
    originalINBreastMasksPath = "../INbreast/original/AllXML"
    INBreastCSVPath = "../INbreast/original/INbreast.csv"
    completePreprocessedIMGFolderPath = "../INbreast/original/INBREAST-Original-Preprocessed-IMG"
    completePreprocessedMSKFolderPath = "../INbreast/original/INBREAST-Original-Preprocessed-MSK"

    INBreastPreprocessing(imagesDirPath = originalINBreastImagesPath,
                          masksDirPath = originalINBreastMasksPath,
                          csvPath = INBreastCSVPath,
                          outputImagesPath = completePreprocessedIMGFolderPath,
                          outputMasksPath = completePreprocessedMSKFolderPath)

    """CropAndSaveImagesRoutine(MassImagesPath = "../INbreast/INBREAST-SELECTED-IMGS",
                             MassMasksPath = "../INbreast/INBREAST-SELECTED-MSKS",
                             croppedMassImagesPath = "../INbreast/INBREAST-Cropped-IMG",
                             croppedMassMasksPath = "../INbreast/INBREAST-Cropped-MSK")"""


# TODO Check this, should process directly xmls or have a preprocessing step (in INbreast module)
#  where xmls are converted into png files ?
def INBreastPreprocessing(imagesDirPath, masksDirPath, csvPath, outputImagesPath, outputMasksPath):
    # Get individual .dcm paths.
    dcmPaths = []

    # Check whether the specified i exists or not
    isExist = os.path.exists(outputImagesPath)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(outputImagesPath)

    # Check whether the specified i exists or not
    isExist = os.path.exists(outputMasksPath)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(outputMasksPath)

    for currentDirectory, dirs, files in os.walk(imagesDirPath):
        files.sort()
        for f in files:
            if f.endswith(".dcm"):
                dcmPaths.append(os.path.join(currentDirectory, f))

    # Get individual xml paths.
    xmlPaths = []

    for currentDirectory, dirs, files in os.walk(masksDirPath):
        files.sort()
        for f in files:
            if f.endswith(".xml"):
                xmlPaths.append(os.path.join(currentDirectory, f))

    # TODO place this into preprocessing of InBreast ??
    dcmPaths, xmlPaths = removeBenignAcquisitions(dicomFilenames = dcmPaths,
                                                  csvFilePath = csvPath,
                                                  xmlFiles = xmlPaths)
    count = 0
    for fullMammPath in dcmPaths:

        imgID = fullMammPath.split('/')[-1]
        imgID = imgID.split('_')[0]

        mp = [i for i in xmlPaths if imgID in i]

        if len(mp) == 0:
            continue

        mp = mp[0]

        # Read full mammogram .dcm file.
        ds = pydicom.dcmread(fullMammPath)

        fullMamm = ds.pixel_array
        print(fullMammPath)

        # =========================
        # Preprocess Full Mammogram
        # =========================

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
        outputFormat = ".png"

        # Preprocess full mammogram images.
        fullMammPreprocessed, toLRFlip = fullMammoPreprocess(
                logger = None,
                img = fullMamm,
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
                mode = 'cbis'
        )

        # Need to normalise to [0, 255] before saving as .png.
        """fullMammPreprocessedNormalized = cv2.normalize(
                fullMammPreprocessed,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )"""

        # Read mask .xml file
        mShape = (ds.pixel_array.shape[0], ds.pixel_array.shape[1])
        maskArray = loadInbreastMask(mask_path = mp, filter = True, imshape = mShape)

        if np.sum(maskArray) < 6000:
            continue

        # Save preprocessed full mammogram image.
        savedFilename = (
                os.path.basename(fullMammPath).replace(".dcm", "")
                + "___PRE"
                + outputFormat
        )
        savedFilePath = os.path.join(outputImagesPath, savedFilename)

        cv2.imwrite(savedFilePath, fullMammPreprocessed)
        # print(f"DONE FULL: {fullMammPath}")

        # ================================
        # Preprocess Corresponding Mask(s)
        # ================================

        # Preprocess.
        mask_pre = maskPreprocess(logger = None, mask = maskArray, toLRFlip = toLRFlip)

        mask_pre = cv2.normalize(
                mask_pre,
                None,
                alpha = 0,
                beta = 255,
                norm_type = cv2.NORM_MINMAX,
                dtype = cv2.CV_32F,
        )

        # Save preprocessed mask.
        savedFilename = (
                os.path.basename(mp).replace(".xml", "") + "___PRE" + outputFormat
        )
        savedFilePath = os.path.join(outputMasksPath, savedFilename)
        cv2.imwrite(savedFilePath, mask_pre)

        #  print(f"DONE MASK: {mp}")

        count += 1

        # if count == 1:
        #     break

    print(f"Total count = {count}")
    print()
    print("Getting out of imagePreprocessing module.")
    print("-" * 30)

    return


def BCDRFullRoutine():
    originalBCDRImagesPath = "../BCDR/BCDR-Images-Original"
    originalBCDRMasksPath = "../BCDR/BCDR-Masks-Original"
    completePreprocessedIMGFolderPath = "../BCDR/BCDR-Original-Preprocessed-IMG"
    completePreprocessedMSKFolderPath = "../BCDR/BCDR-Original-Preprocessed-MSK"
    BCDRPreprocessing(imagesDirPath = originalBCDRImagesPath,
                      masksDirPath = originalBCDRMasksPath,
                      outputImagesPath = completePreprocessedIMGFolderPath,
                      outputMasksPath = completePreprocessedMSKFolderPath)
    CropAndSaveImagesRoutine(MassImagesPath = "../BCDR/BCDR-SELECTED-IMGS",
                             MassMasksPath = "../BCDR/BCDR-SELECTED-MASKS",
                             croppedMassImagesPath = "../BCDR/BCDR-Cropped-FINAL-IMG",
                             croppedMassMasksPath = "../BCDR/BCDR-Cropped-FINAL-MSK")


if __name__ == '__main__':
    # CSAWFullRoutine()
    CBISFullRoutine()
    # INBreastFullRoutine()
    # BCDRFullRoutine()
    pass
