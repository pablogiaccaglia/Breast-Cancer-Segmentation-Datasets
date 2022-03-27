# from logs import logDecorator as lD
import jsonref, os
import numpy as np
import cv2

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.overlayMasks.overlayMasks"

configOverlay = jsonref.load(open("../config/modules/overlayMasks.json"))


# @lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for overlayMasks.
    This function overlays a given ground truth mask with its
    corresponding predicted mask.
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    print("=" * 30)
    print("Main function of overlayMasks.")
    print("=" * 30)

    # Get parameters from .json files.
    fullImagesDirectory = configOverlay["fullImagesDirectory"]
    trueMasksDirectory = configOverlay["trueMasksDirectory"]
    predictedMasksDirectory = configOverlay["predictedMasksDirectory"]
    extension = configOverlay["extension"]
    imagesTargetSize = (configOverlay["imagesTargetSize"], configOverlay["imagesTargetSize"])
    overlayMasksDirectory = configOverlay["overlayMasksDirectory"]
    segmentedImagesDirectory = configOverlay["segmentedImagesDirectory"]

    # ------------

    # Get paths.
    fullImagesPathsList = []
    trueMasksPathsList = []
    predictedMasksPathsList = []

    for full in os.listdir(fullImagesDirectory):
        if full.endswith(extension):
            fullImagesPathsList.append(os.path.join(fullImagesDirectory, full))

    for full in os.listdir(trueMasksDirectory):
        if full.endswith(extension):
            trueMasksPathsList.append(os.path.join(trueMasksDirectory, full))

    for full in os.listdir(predictedMasksDirectory):
        if full.endswith(extension):
            predictedMasksPathsList.append(os.path.join(predictedMasksDirectory, full))

    fullImagesPathsList.sort()
    trueMasksPathsList.sort()
    predictedMasksPathsList.sort()

    # ------------

    # Load fullImages
    fullImagesArray = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=imagesTargetSize)
        for path in fullImagesPathsList
    ]

    # Load true masks.
    trueMasksArray = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=imagesTargetSize)
        for path in trueMasksPathsList
    ]

    # Load predicted masks.
    predictedMasksArray = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=imagesTargetSize)
        for path in predictedMasksPathsList
    ]

    print(fullImagesArray[0].min(), fullImagesArray[0].max())
    print(trueMasksArray[0].min(), trueMasksArray[0].max())
    print(predictedMasksArray[0].min(), predictedMasksArray[0].max())

    # ------------

    # Stack to create RGB version of grayscale images.
    fullImagesRGBArray = [np.stack([img, img, img], axis=-1) for img in fullImagesArray]

    # Green true mask. Note OpenCV uses BGR.
    trueMasksRGBArray = [
        np.stack([np.zeros_like(img), img, np.zeros_like(img)], axis=-1)
        for img in trueMasksArray
    ]

    # Red predicted mask. Note OpenCV uses BGR.
    predictedMasksRGBArray = [
        np.stack([np.zeros_like(img), np.zeros_like(img), img], axis=-1)
        for img in predictedMasksArray
    ]

    # ------------

    for i in range(len(fullImagesRGBArray)):

        # First overlay true and predicted masks.
        overlayMasks = cv2.addWeighted(
            src1=trueMasksRGBArray[i], alpha=0.5, src2=predictedMasksRGBArray[i], beta=1, gamma=0
        )

        # Then overlay full images and masks.
        segmentedImages = cv2.addWeighted(
            src1=fullImagesRGBArray[i], alpha=1, src2=overlayMasks, beta=0.5, gamma=0
        )

        # Save.

        # Get patient ID from true masks.
        filename = os.path.basename(trueMasksPathsList[i])
        filenameWords = filename.split("_")
        patientID = "_".join([filenameWords[i] for i in range(4)])

        overlayMasksFilename = patientID + "___MasksOverlay.png"
        segmentedImagesFilename = patientID + "___AllOverlay.png"

        overlayMasksPath = os.path.join(overlayMasksDirectory, overlayMasksFilename)
        segmentedImagesPath = os.path.join(segmentedImagesDirectory, segmentedImagesFilename)

        print(overlayMasksPath)
        print(segmentedImagesPath)

        cv2.imwrite(filename=overlayMasksPath, img=overlayMasks)
        cv2.imwrite(filename=segmentedImagesPath, img=segmentedImages)