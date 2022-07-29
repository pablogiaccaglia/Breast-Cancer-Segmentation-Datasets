import os
import cv2
import os
from typing import Union

import cv2
import numpy as np

from preprocessing import *
path1 = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-MASS-FINAL-IMG/"
path2 = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-MASS-FINAL-MSK/"

s1 = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-MASS-EXPERIMENT-IMG/"
s2 = "/Users/pablo/Desktop/nl2-project/CBIS/CBSI-MASS-EXPERIMENT-MSK/"


entries1 = os.listdir(s1)
entries1.sort()

entries2 = os.listdir(s2)
entries2.sort()

print(len(entries1))
print(len(entries2))


def fullMammoPreprocess2(
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
) -> np.ndarray:
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
        # cv2.imwrite("../data/preprocessed/Mass/testing/cropped.png", croppedImage)

        # Step 2: Min-max normalize.
        normalizedImage = minMaxNormalize(logger = logger, img = img)
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

            maskedImage = applyMask(logger = logger, img = img, mask = xLargestMask)
            # cv2.imwrite("../data/preprocessed/Mass/testing/maskedImage.png", maskedImage)

    except Exception as e:
        # logger.error(f'Unable to fullMammPreprocess!\n{e}')
        print(f"Unable to fullMammPreprocess!\n{e}")

    return maskedImage


for im in entries1:
    if im.endswith('.png'):
        imarr = cv2.imread(s1+im, cv2.cv2.IMREAD_GRAYSCALE)

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
        try:
            fullMammPreprocessed = fullMammoPreprocess2(
                    logger = None,
                    img = imarr,
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
        except:
            print(path1 + im)

cv2.imwrite(path1 + im, fullMammPreprocessed)


