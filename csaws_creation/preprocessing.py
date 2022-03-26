from typing import Union

import cv2
import numpy as np
import albumentations as A


def cropBorders(logger, img,
                left: Union[float, int] = 0.01,
                right: Union[float, int] = 0.01,
                up: Union[float, int] = 0.04,
                down: Union[float, int] = 0.04) -> np.ndarray:
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

def globalBinarize(logger, img: np.ndarray, thresh, maxval: np.uint8) -> np.ndarray:
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

def editMask(logger, mask: np.ndarray, ksize: tuple = (23, 23), operation: str = "open") -> np.ndarray:
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
        colSum = mask.sum(axis=0)

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

def pad(logger, img: np.ndarray) -> np.ndarray:

    """
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.
    If the image is portrait, black pixels will be
    padded on the right to form a square.
    If the image is landscape, black pixels will be
    padded on the bottom to form a square.
    Parameters
    ----------
    img : {numpy.ndarray}
        The image to pad.
    Returns
    -------
    paddedImage : {numpy.ndarray}
        The padded square image, if padding was required
        and done.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    """

    paddedImg = img

    try:

        nRows, nCols = img.shape

        # If padding is required...
        if nRows != nCols:
            # Take the longer side as the target shape.
            if nCols < nRows:
                targetShape = (nRows, nRows)
            elif nRows < nCols:
                targetShape = (nCols, nCols)

            # pad.
            padded_img = np.zeros(shape= targetShape)
            padded_img[:nRows, :nCols] = img

    except Exception as e:
        # logger.error(f'Unable to pad!\n{e}')
        print(f"Unable to pad!\n{e}")

    return paddedImg

def CLAHE(logger, image: np.ndarray) -> dict:

    augmented = image

    try:
        aug = A.CLAHE(p = 1)
        augmented = aug(image = image)
    except Exception as e:
        # logger.error(f'Unable to apply CLAHE!\n{e}')
        print(f'Unable to apply CLAHE!\n{e}')

    return augmented


def fullMammoPreprocess(
    logger,
    img: np.ndarray,
    left : Union[float, int],
    right : Union[float, int],
    down : Union[float, int],
    up : Union[float, int],
    thresh,
    maxval: np.uint8,
    ksize: tuple,
    operation: str,
    reverse: bool,
    topXContours: int
):

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

    imgPreprocessed, toLRFlip = img, False

    try:
        # Step 1: Initial crop.
        croppedImage = cropBorders(logger = logger, img=img, left=left, right=right, down=down, up=up)
        # cv2.imwrite("../data/preprocessed/Mass/testing/cropped.png", croppedImage)

        # Step 2: Min-max normalize.
        normalizedImage = minMaxNormalize(logger = logger, img=croppedImage)
        # cv2.imwrite("../data/preprocessed/Mass/testing/normed.png", normalizedImage)

        # Step 3: Remove artefacts.
        binarizedImage = globalBinarize(logger = logger, img=normalizedImage, thresh=thresh, maxval=maxval)
        editedMask = editMask(
                logger = logger, mask=binarizedImage, ksize=(ksize, ksize), operation=operation
        )
        _, xLargestMasks = selectXLargestBlobs(logger = logger, mask=editedMask, topXContours =topXContours, reverse=reverse)
        # cv2.imwrite(
        # "../data/preprocessed/Mass/testing/xLargest_mask.png", xLargestMasks
        # )
        maskedImage = applyMask(logger = logger, img=normalizedImage, mask=xLargestMasks)
        # cv2.imwrite("../data/preprocessed/Mass/testing/maskedImage.png", maskedImage)

        # Step 4: Horizontal flip.
        toLRFlip = checkLRFlip(logger = logger, mask=xLargestMasks)

        if toLRFlip:
            flippedImage = makeLRFlip(logger = logger, img=maskedImage)
        elif not toLRFlip:
            flippedImage = maskedImage
        # cv2.imwrite("../data/preprocessed/Mass/testing/flippedImage.png", flippedImage)

        # Step 5: CLAHE enhancement.
        claheImage = CLAHE(logger = logger, image=flippedImage)['image']
        # cv2.imwrite("../data/preprocessed/Mass/testing/claheImage.png", claheImage)

        # Step 6: pad.
        paddedImage = pad(logger = logger, img=claheImage)
        paddedImage = cv2.normalize(
            paddedImage,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        # cv2.imwrite("../data/preprocessed/Mass/testing/paddedImage.png", paddedImage)

        # Step 7: Downsample.
        # Not done yet.

        # Step 8: Min-max normalise.
        imgPreprocessed = minMaxNormalize(logger = logger, img=paddedImage)
        # cv2.imwrite("../data/preprocessed/Mass/testing/imgPreprocessed.png", imgPreprocessed)

    except Exception as e:
        # logger.error(f'Unable to fullMammPreprocess!\n{e}')
        print(f"Unable to fullMammPreprocess!\n{e}")

    return imgPreprocessed, toLRFlip

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
    mask = cropBorders(logger, img=mask)

    # Step 2: Horizontal flip.
    if toLRFlip:
        mask = makeLRFlip(logger, img=mask)

    # Step 3: Pad.
    maskPreprocessed = pad(logger, img=mask)

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
            src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
        )

    except Exception as e:
        # logger.error(f'Unable to findMultiTumour!\n{e}')
        print((f"Unable to get findMultiTumour!\n{e}"))

    return
