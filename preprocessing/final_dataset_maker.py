import os
from itertools import combinations
import random
from typing import Union
import re
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import shutil
from augmentations import augment
from augmentations import rotateFlipData
from preprocessing import getMaskPatch, getPatches


def imgAugment(logger, self, x_img, y_img):
    """
    Follows the flow:
    1.         datasetPaths()
             _______|_______
            |               |
    2. loadFullImg()  loadMaskImg()
            |_______________|
                    |
    3.          tfParse()
                    |
    4.         imgAugment() *
                    |
    5.        makeTFDataset()

    Apply random image augmentation to full mammogram scans (x_img). Note
    that the same augmentation has to be applied to the masks (y_img), apart
    from the brightness augmentation.

    Parameters
    ----------
    x_img : {numpy.ndarray}
        Full mammogram scan to augment.
    y_img : {numpy.ndarray}
        Corresponding mask of `x_img`.

    Returns
    -------
    x_img : {numpy.ndarray}
        Augmented x_img.
    y_img : {numpy.ndarray}
        Augmented y_img.
    """

    try:

        aug_x, aug_y = augment(image = x_img, mask = y_img)

    except Exception as e:
        # logger.error(f'Unable to imgAugment!\n{e}')
        print(f"Unable to imgAugment!\n{e}")

    return x_img, y_img


def datasetPaths(
        logger,
        full_img_dir,
        mask_img_dir,
        extension,
):
    """
        Follows the flow:
        1.         datasetPaths() *
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the directories of the folder containing the full mammogram
        scans (x) and the ground truth masks (y) and returns a list of paths to
        individual full scans and masks.

        Parameters
        ----------
        full_img_dir : {str}
            Directory that contains the FULL training images.
        mask_img_dir : {str}
            Directory that contains the MASK training images.
        extension : {str}
            The file extension of the images (e.g. ".png")

        Returns
        -------
        x_paths_list: {list}
            A list of paths to individual FULL images in the training set.
        y_paths_list: {list}
            A list of paths to individual MASK images in the training set.
        """

    x_paths_list = []
    y_paths_list = []

    full_img_dir = full_img_dir
    mask_img_dir = mask_img_dir

    try:

        # =======================================
        #  1. Get paths of X (full) and y (mask)
        # =======================================

        if type(full_img_dir) != list and type(mask_img_dir):
            full_img_dir = [full_img_dir]
            mask_img_dir = [mask_img_dir]

        for dir in full_img_dir:
            # Get paths of train images and masks.
            for full in os.listdir(dir):
                if full.endswith(extension):
                    x_paths_list.append(os.path.join(dir, full))

        for dir in mask_img_dir:
            for mask in os.listdir(dir):
                if mask.endswith(extension):
                    y_paths_list.append(os.path.join(dir, mask))

        # ** IMPORTANT ** Sort so that FULL and MASK images are in an order
        # that corresponds with each other.
        x_paths_list.sort()
        y_paths_list.sort()

    except Exception as e:
        # logger.error(f'Unable to datasetPaths!\n{e}')
        print(f"Unable to datasetPaths!\n{e}")

    return x_paths_list, y_paths_list


def loadImg(logger, i, m = None, mode: str = None, patchify: bool = False, getPatchMask = False,
            overlappingPatches: bool = False):
    """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()* loadMaskImg()
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the m of a FULL image and loads it as a grayscale image (as
        a numpy array with 3 channels).

        Parameters
        ----------
        i : {str}
            The m of the FULL image.

        Returns
        -------
        fullImg : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 3)
        """

    try:
        # =====================================
        #  2a. Read images (full)
        # =====================================
        if isinstance(i, str):
            img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
            img = img.astype(np.float32)
        elif isinstance(i, np.ndarray):
            img = i

        patchedMaskImage = None
        coords = None

        if mode == 'net_data':
            if getPatchMask and m:
                if isinstance(m, str):
                    msk = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                elif isinstance(m, np.ndarray):
                    msk = m
                patchedMaskImage, coords = getMaskPatch(img, msk)

                # TODO handle better this case, in which the image doesn't contains no mask!!!
                if patchedMaskImage.shape[0] != 0 or patchedMaskImage.shape[1] != 0:
                    patchedMaskImage = cv2.resize(patchedMaskImage, (256, 256))

                else:
                    # print(i)
                    patchedMaskImage = None
                    coords = None

            """width = img.shape[0]  # 3757
            param = 5
            dim = width // param  # 751
            multiplier = dim // 256
            fixedDim = 256 * multiplier"""

            fixedDim = 2048
            if overlappingPatches:
                step = 1024
            else:
                step = fixedDim

            # 1005 x 2483 = 2.495.415

            f = lambda x: cv2.resize(x, (256, 256))

            if patchify:
                try:
                    patches, shape = getPatches(img, width = fixedDim, height = fixedDim, channels = None, step = step)
                    patches = list(map(f, patches))
                except:

                    try:
                        # print(i)
                        fixedDim = 1024

                        if overlappingPatches:
                            step = 512
                        else:
                            step = fixedDim
                        patches, shape = getPatches(img, width = fixedDim, height = fixedDim, channels = None,
                                                    step = step)
                        patches = list(map(f, patches))
                    except:
                        # print(i)
                        fixedDim = 512
                        if overlappingPatches:
                            step = 256
                        else:
                            step = fixedDim
                        patches, shape = getPatches(img, width = fixedDim, height = fixedDim, channels = None,
                                                    step = step)
                        patches = list(map(f, patches))

                if patchedMaskImage is not None:
                    patches.append(patchedMaskImage)
                if getPatchMask:
                    return patches, coords
                else:
                    return patches, fixedDim, shape
            else:
                if getPatchMask:
                    return patchedMaskImage, coords
                else:
                    return patchedMaskImage

        else:
            try:
                img = cv2.resize(img, (256, 256, 3))
            except:
                img = cv2.resize(img, (256, 256))
            return img

    except Exception as e:
        # logger.error(f'Unable to loadFullImg!\n{e}')
        print(f"Unable to loadFullImg!\n{e}")
        print(i)


def loadMaskImg(logger, m, mode: str, coords = None, patchify = False, overlappingPatches: bool = False):
    """
        Follows the flow:
        1.         datasetPaths()
                 _______|_______
                |               |
        2. loadFullImg()  loadMaskImg() *
                |_______________|
                        |
        3.          tfParse()
                        |
        4.         imgAugment()
                        |
        5.        makeTFDataset()

        Takes in the i of a MASK image and loads it as a grayscale image (as
        a numpy array with 1 channel).

        Parameters
        ----------
        m : {str}
            The i of the MASK image.

        Returns
        -------
        full_img : {numpy.ndarray}
            The loaded image with shape = (self.target_size, self.target_size, 1)
        """

    try:

        # ========================
        #  2b. Read images (mask)
        # ========================
        if isinstance(m, str):
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
        elif isinstance(m, np.ndarray):
            mask = m

        patchedMask = mask  # (x,y,1)

        if coords is not None:
            patchedMask = patchedMask[coords[0]: coords[1], coords[2]: coords[3]]
            patchedMask = cv2.resize(patchedMask, (256, 256))
        else:
            patchedMask = None

        fixedDim = 2048
        if overlappingPatches:
            step = 1024
        else:
            step = fixedDim

        # 1005 x 2483 = 2.495.415

        f = lambda x: cv2.resize(x, (256, 256))

        if patchify:

            try:
                patches, shape = getPatches(mask, width = fixedDim, height = fixedDim, channels = None, step = step)
                patches = list(map(f, patches))
            except:

                try:
                    # print(i)
                    fixedDim = 1024

                    if overlappingPatches:
                        step = 512
                    else:
                        step = fixedDim

                    patches, shape = getPatches(mask, width = fixedDim, height = fixedDim, channels = None, step = step)
                    patches = list(map(f, patches))
                except:
                    # print(i)
                    fixedDim = 512
                    if overlappingPatches:
                        step = 256
                    else:
                        step = fixedDim
                    patches, shape = getPatches(mask, width = fixedDim, height = fixedDim, channels = None, step = step)
                    patches = list(map(f, patches))

            if patchedMask is not None:
                patches.append(patchedMask)

            return patches, fixedDim, shape

        else:
            if patchedMask is not None:
                mask = patchedMask

        return mask

    except Exception as e:
        # logger.error(f'Unable to loadMaskImg!\n{e}')
        print(f"Unable to loadMaskImg!\n{e}")


def checkAllDifferent(elems: list):
    if not len(elems):
        return False

    for pair in combinations(elems, 2):
        if np.array_equal(pair[0], pair[1]):
            return False

    return True


def augmentDataset(images: list, masks: list, augmentingFactor: int) -> (list, list):
    augmented_test_images = []
    augmented_test_masks = []
    toCheck = []
    created = []

    for img, msk in zip(images, masks):
        created.clear()
        toCheck.clear()

        while not checkAllDifferent(elems = toCheck):
            created.clear()
            toCheck.clear()
            toCheck.append(img)

            for _ in range(augmentingFactor):
                aug_x, aug_y = augment(image = img, mask = msk)
                created.append((aug_x, aug_y))
                toCheck.append(aug_x)

        for aug_x, aug_y in created:
            augmented_test_images.append(aug_x)
            augmented_test_masks.append(aug_y)

    return augmented_test_images, augmented_test_masks


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def loadData(imagesPath: Union[str, list[str]], masksPath: Union[str, list[str]], mode: str,
             augmentationFactor = None, req: int = None):
    # ====================
    #  Create test images
    # ====================
    # Get paths to individual images.
    test_x, test_y = datasetPaths(
            logger = None,
            full_img_dir = imagesPath,
            mask_img_dir = masksPath,
            extension = ".png",
    )

    c = list(zip(test_x, test_y))

    random.shuffle(c)

    test_x, test_y = zip(*c)
    # Read FULL images.

    imgs = []
    masks = []

    if mode == 'net_data':

        coords = []
        patchify = True
        overlappingPatches = True

        for path, mskPath in zip(test_x, test_y):
            img, coord = loadImg(logger = None, i = path, m = mskPath, mode = mode,
                                 patchify = patchify, overlappingPatches = overlappingPatches, getPatchMask = True)

            if patchify:  # img is a list
                imgs = imgs + img

            else:  # img is numpy array
                imgs.append(img)

            coords.append(coord)

        # Read MASK images.

        print(len(coords))
        print(len(test_y))
        print(len(test_x))
        print(len(imgs))
        print(len(masks))

        for x, path, c in zip(test_x, test_y, coords):
            msk, _, _ = loadMaskImg(logger = None, m = path, mode = mode, coords = c,
                                    patchify = patchify, overlappingPatches = overlappingPatches)

            if patchify:
                masks = masks + msk  # msk is a list
            else:
                masks.append(msk)  # msk is numpy array

        # cleaning part, here we first move images and masks containing masses or portions of it in a separated list,
        # then we filter the remaining pairs such as 40% of the whole paris contains masses, 60% doesn't contain them.
        # Of this 60%, 10% are totally black patches. This is a reasonable choice since there are a lot of patches
        # which are not completely black which are not in this 10%.

        print(len(imgs))
        print(len(masks))

        if patchify:
            imagesWithMasses = []
            masksWithMasses = []

            imagesWithoutMasses = []
            masksWithoutMasses = []

            for im, mask in zip(imgs, masks):
                if mask.sum() > 0:
                    imagesWithMasses.append(im)
                    masksWithMasses.append(mask)
                else:
                    imagesWithoutMasses.append(im)
                    masksWithoutMasses.append(mask)

            # del imgs
            # del masks

            lenImagesWithMasses = len(imagesWithMasses)
            lenImagesWithoutMasses = int(0.5 * (lenImagesWithMasses / 0.5))
            lenBlackImages = int(lenImagesWithoutMasses * 0.1)
            lenNotBlackImages = lenImagesWithoutMasses - lenBlackImages

            keptImagesWithoutMasses = []
            keptMasksWithoutMasses = []

            counterBlackImages = 0
            counterNotBlackImages = 0

            # some shuffling

            c = list(zip(imagesWithoutMasses, masksWithoutMasses))

            random.shuffle(c)

            imagesWithoutMasses, masksWithoutMasses = zip(*c)

            for im, mask in zip(imagesWithoutMasses, masksWithoutMasses):

                if im.sum() > 0:

                    if counterNotBlackImages >= lenNotBlackImages:
                        continue
                    counterNotBlackImages += 1

                else:

                    if counterBlackImages >= lenBlackImages:
                        continue
                    counterBlackImages += 1

                keptImagesWithoutMasses.append(im)
                keptMasksWithoutMasses.append(mask)

            imagesWithoutMasses = keptImagesWithoutMasses
            masksWithoutMasses = keptMasksWithoutMasses

            print(len(imagesWithMasses))
            print(len(masksWithMasses))
            print(len(imagesWithoutMasses))
            print(len(masksWithoutMasses))

            imgs = imagesWithoutMasses + imagesWithMasses
            masks = masksWithoutMasses + masksWithMasses

    elif mode == 'ndarrays':

        # Read FULL images.
        imgs = [
            loadImg(logger = None, i = path, mode = mode)
            for path in test_x
        ]

        # Read MASK images.
        masks = [
            loadMaskImg(logger = None, m = path, mode = mode)
            for path in test_y]

    if mode == "augment" and augmentationFactor is not None:

        augmented_images, augmented_masks = augmentDataset(images = imgs,
                                                           masks = masks,
                                                           augmentingFactor = augmentationFactor)

        tempImgs = []
        tempMsks = []

        for img, msk in zip(imgs, masks):
            img, msk = rotateFlipData(image = img, mask = msk)
            tempImgs.append(img)
            tempMsks.append(msk)

        imgs = tempImgs
        masks = tempMsks

        masks = masks + augmented_masks
        imgs = imgs + augmented_images

    elif mode == "net_data":
        pass

    return imgs, masks


def saveImages(images: list[np.ndarray], dirPath: str, outputFormat: str, baseName: str, prefix: str):
    for i in range(len(images)):
        # Save preprocessed full mammogram image.
        savedFilePath = dirPath + prefix + baseName + "__" + str(i + 1).zfill(5) + outputFormat
        cv2.imwrite(savedFilePath, images[i])


def prepareData(imagesPath: Union[str, list[str]], masksPath: Union[str, list[str]], mode: str,
                augmentingFactor: int = None, req: int = None) -> tuple[list, list]:
    imgs, masks = loadData(imagesPath = imagesPath,
                           masksPath = masksPath,
                           augmentationFactor = augmentingFactor, mode = mode, req = req)

    return imgs, masks


def rflip(a, b):
    x = []
    y = []

    for img, msk in zip(a, b):
        img, msk = rotateFlipData(image = img, mask = msk)
        x.append(img)
        y.append(msk)

    return x, y


def makeFinalDataset(imagesDir: Union[str, list[str]],
                     masksDir: Union[str, list[str]],
                     trainingImgDir: str,
                     trainingMaskDir: str,
                     validationImgDir: str,
                     validationMaskDir: str,
                     testingImgDir: str,
                     testingMaskDir: str,
                     augmentingFactor: int,
                     trainRatio: float,
                     validationRatio: float,
                     testRatio: float,
                     outputFormat: str,
                     prefix: str):
    imgs, imgsMasks, = prepareData(imagesPath = imagesDir,
                                   masksPath = masksDir,
                                   augmentingFactor = augmentingFactor, mode = 'net_data')

    """    plt.imshow(imgs[0], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[1], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[2], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[3], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[4], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[5], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[6], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[7], cmap = plt.cm.gray)
        plt.show()
    
        plt.imshow(imgs[8], cmap = plt.cm.gray)
        plt.show() """

    xTrain, xTest, yTrain, yTest = train_test_split(imgs, imgsMasks, test_size = 1 - trainRatio)

    """ for i in range(len(xTest)):
        if yTest[i].sum() > 0:
            plt.imshow(xTest[i], cmap = plt.cm.gray)
            plt.show()

            plt.imshow(yTest[i], cmap = plt.cm.gray)
             plt.show()
    """
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    xVal, xTest, yVal, yTest = train_test_split(xTest, yTest,
                                                test_size = testRatio / (testRatio + validationRatio), shuffle = False)

    augmentedXTrainingImgs, augmentedYTrainingMsks = augmentDataset(images = xTrain,
                                                                    masks = yTrain,
                                                                    augmentingFactor = augmentingFactor)

    augmentedXTestingImgs, augmentedYTestingMsks = augmentDataset(images = xTest,
                                                                  masks = yTest,
                                                                  augmentingFactor = augmentingFactor)

    augmentedXValidationImgs, augmentedYValidationMsks = augmentDataset(images = xVal,
                                                                        masks = yVal,
                                                                        augmentingFactor = augmentingFactor)

    xTrain, yTrain = rflip(a = xTrain, b = yTrain)

    yTrain = yTrain + augmentedYTrainingMsks
    xTrain = xTrain + augmentedXTrainingImgs

    xVal, yVal = rflip(a = xVal, b = yVal)

    yVal = yVal + augmentedYValidationMsks
    xVal = xVal + augmentedXValidationImgs

    xTest, yTest = rflip(a = xTest, b = yTest)

    yTest = yTest + augmentedYTestingMsks
    xTest = xTest + augmentedXTestingImgs

    print("ciao")

    saveImages(images = xTrain, dirPath = trainingImgDir, outputFormat = outputFormat, baseName = "Training-IMG",
               prefix = prefix)
    saveImages(images = yTrain, dirPath = trainingMaskDir, outputFormat = outputFormat, baseName = "Training-MSK",
               prefix = prefix)

    saveImages(images = xVal, dirPath = validationImgDir, outputFormat = outputFormat, baseName = "Validation-IMG",
               prefix = prefix)
    saveImages(images = yVal, dirPath = validationMaskDir, outputFormat = outputFormat, baseName = "Validation-MSK",
               prefix = prefix)

    saveImages(images = xTest, dirPath = testingImgDir, outputFormat = outputFormat, baseName = "Testing-IMG",
               prefix = prefix)
    saveImages(images = yTest, dirPath = testingMaskDir, outputFormat = outputFormat, baseName = "Testing-MSK",
               prefix = prefix)


def routineMakeFinalDataset():
    imagesMassPathCBIS = "../CBIS/cropped/CBIS-MASS-Cropped-FINAL-IMG"
    masksMassPathCBIS = "../CBIS/cropped/CBIS-MASS-Cropped-FINAL-MSK"

    imagesCalcPathCBIS = "../CBIS/CBIS-Original-Calc-Preprocessed-Complete-IMG"
    masksCalcPathCBIS = "../CBIS/CBIS-Original-Calc-Preprocessed-Complete-MSK"

    imagesPathCSAW = "../CSAW/CSAW-Cropped-FINAL-IMG"
    masksPathCSAW = "../CSAW/CSAW-Cropped-FINAL-MSK"

    imagesPathBCDR = "../BCDR/BCDR-Cropped-FINAL-IMG"
    masksPathBCDR = "../BCDR/BCDR-Cropped-FINAL-MSK"

    imagesPathINBreast = "../BCDR/BCDR-Cropped-FINAL-IMG"
    masksPathINBreast = "../BCDR/BCDR-Cropped-FINAL-MSK"

    imagesPathCDD = "../CDD/PKG - CDD-CESM/Low energy images of CDD-CESM"
    masksPathCDD = "../CDD/real_segmentations"

    trainingImgDir = "../splits/Dataset-split/CBIS-Training-Final-IMG/"
    trainingMaskDir = "../splits/Dataset-split/CBIS-Training-Final-MSK/"
    validationImgDir = "../splits/Dataset-split/CBIS-Validation-Final-IMG/"
    validationMaskDir = "../splits/Dataset-split/CBIS-Validation-Final-MSK/"
    testingImgDir = "../splits/Dataset-split/CBIS-Testing-Final-IMG/"
    testingMaskDir = "../splits/Dataset-split/CBIS-Testing-Final-MSK/"

    for dir in [trainingImgDir, trainingMaskDir,
                validationImgDir, validationMaskDir,
                testingImgDir, testingMaskDir]:
        try:
            shutil.rmtree(dir)
        except:
            pass
        os.makedirs(dir)

    trainRatio = 0.75
    validationRatio = 0.15
    testRatio = 0.10

    augmentingFactor = 2

    outputFormat = '.png'

    makeFinalDataset(imagesDir = [imagesMassPathCBIS],
                     masksDir = [masksMassPathCBIS],
                     trainingImgDir = trainingImgDir,
                     trainingMaskDir = trainingMaskDir,
                     validationImgDir = validationImgDir,
                     validationMaskDir = validationMaskDir,
                     testingImgDir = testingImgDir,
                     testingMaskDir = testingMaskDir,
                     augmentingFactor = augmentingFactor,
                     trainRatio = trainRatio,
                     validationRatio = validationRatio,
                     testRatio = testRatio,
                     outputFormat = outputFormat, prefix = 'final-')


def _getDatasets(trainingImgDir: str,
                 trainingMaskDir: str,
                 validationImgDir: str,
                 validationMaskDir: str,
                 testingImgDir: str,
                 testingMaskDir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainingImgs, trainingMasks = prepareData(imagesPath = trainingImgDir,
                                              masksPath = trainingMaskDir,
                                              augmentingFactor = None, mode = 'ndarrays', req = 8000)
    validationImgs, validationMasks = prepareData(imagesPath = validationImgDir,
                                                  masksPath = validationMaskDir,
                                                  augmentingFactor = None, mode = 'ndarrays', req = 1400)
    testingImgs, testingMasks = prepareData(imagesPath = testingImgDir,
                                            masksPath = testingMaskDir,
                                            augmentingFactor = None, mode = 'ndarrays', req = 1600)

    print(trainingImgs[0].shape)
    print(trainingMasks[1].shape)

    return np.stack(trainingImgs), np.stack(trainingMasks), np.stack(validationImgs), np.stack(
            validationMasks), np.stack(testingImgs), np.stack(testingMasks)


def getDatasetForNet() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainingImgDir = "../splits/Dataset-split/CBIS-Training-Final-IMG/"
    trainingMaskDir = "../splits/Dataset-split/CBIS-Training-Final-MSK/"
    validationImgDir = "../splits/Dataset-split/CBIS-Validation-Final-IMG/"
    validationMaskDir = "../splits/Dataset-split/CBIS-Validation-Final-MSK/"
    testingImgDir = "../splits/Dataset-split/CBIS-Testing-Final-IMG/"
    testingMaskDir = "../splits/Dataset-split/CBIS-Testing-Final-MSK/"

    return _getDatasets(trainingImgDir = trainingImgDir,
                        trainingMaskDir = trainingMaskDir,
                        validationImgDir = validationImgDir,
                        validationMaskDir = validationMaskDir,
                        testingImgDir = testingImgDir,
                        testingMaskDir = testingMaskDir)


def loadNumpyArrays(folderPath: str, arr) -> np.ndarray:
    i = 0
    for entry in os.scandir(folderPath):

        l = np.load(entry)

        if len(l.shape) == 2:
            l = np.reshape(l, (256, 256, 1))

        print(l.shape)
        arr[i] = l
        i = i + 1
        break

    return arr


def getDatasetArraysForNet() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trainingImgDir = "../CBIS/Dataset-split-arrays/Training-Final-IMG-Arrayss/"
    trainingMaskDir = "../CBIS/Dataset-split-arrays/Training-Final-MSK-Arrays/"
    validationImgDir = "../CBIS/Dataset-split-arrays/Validation-Final-IMG-Arrays"
    validationMaskDir = "../CBIS/Dataset-split-arrays/Testing-Final-MSK-Arrays/"
    testingImgDir = "../CBIS/Dataset-split-arrays/Testing-Final-IMG-Arrays/"
    testingMaskDir = "../CBIS/Dataset-split-arrays/Testing-Final-MSK-Arrays/"

    return loadNumpyArrays(folderPath = trainingImgDir, arr = np.ndarray((13027, 256, 256, 3), dtype = 'float32')), \
           loadNumpyArrays(folderPath = trainingMaskDir, arr = np.ndarray((13027, 256, 256, 1), dtype = 'float32')), \
           loadNumpyArrays(folderPath = validationImgDir,
                           arr = np.ndarray((2605, 256, 256, 3), dtype = 'float32')), loadNumpyArrays(
            folderPath = validationMaskDir,
            arr = np.ndarray((2605, 256, 256, 1), dtype = 'float32')), \
           loadNumpyArrays(folderPath = testingImgDir,
                           arr = np.ndarray((1738, 256, 256, 3), dtype = 'float32')), loadNumpyArrays(
            folderPath = testingMaskDir,
            arr = np.ndarray((1738, 256, 256, 1), dtype = 'float32'))


def Normalize(data, mean_data = None, std_data = None):
    if not mean_data:
        mean_data = np.mean(data)
    if not std_data:
        std_data = np.std(data)
    norm_data = (data - mean_data) / std_data
    return norm_data, mean_data, std_data


def __saveDatasetArraysForNet(folderPath: str, suffix: str, data: np.ndarray):
    for i in range(len(data)):
        index = i + 1
        path = folderPath + suffix + '__' + str(index) + '.npy'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        np.save(file = path, arr = data[i])


def saveDatasetArraysForNet():
    imgs_train, imgs_mask_train, imgs_val, imgs_mask_val, imgs_test, imgs_mask_test_gt = getDatasetForNet()

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train = imgs_train.astype('float32')
    # imgs_train_cbis -= mean
    # imgs_train_cbis /= std
    # imgs_train_cbis /= 255.

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_mask_train = (imgs_mask_train > 0.5).astype(np.uint8)
    imgs_mask_train = imgs_mask_train[..., np.newaxis]

    imgs_test = imgs_test.astype('float32')
    # imgs_test_cbis -= mean
    # imgs_test_cbis /= std
    # imgs_test_cbis /= 255.

    imgs_mask_test_gt = imgs_mask_test_gt.astype('float32')
    imgs_mask_test_gt /= 255.  # scale masks to [0, 1]
    imgs_mask_test_gt = (imgs_mask_test_gt > 0.5).astype(np.uint8)
    imgs_mask_test_gt = imgs_mask_test_gt[..., np.newaxis]

    imgs_val = imgs_val.astype('float32')
    # imgs_val -= mean
    # imgs_val /=std
    # imgs_val /= 255.

    imgs_mask_val = imgs_mask_val.astype('float32')
    imgs_mask_val /= 255.  # scale masks to [0, 1]
    imgs_mask_val = (imgs_mask_val > 0.5).astype(np.uint8)
    imgs_mask_val = imgs_mask_val[..., np.newaxis]

    print("go")

    testingImgArraysDir = "../splits/Dataset-split-arrays/Testing-Final-IMG-Arrays/"
    testingMaskArraysDir = "../splits/Dataset-split-arrays/Testing-Final-MSK-Arrays/"
    validationImgArraysDir = "../splits/Dataset-split-arrays/Validation-Final-IMG-Arrays/"
    validationMaskArraysDir = "../splits/Dataset-split-arrays/Validation-Final-MSK-Arrays/"
    trainingImgArraysDir = "../splits/Dataset-split-arrays/Training-Final-IMG-Arrayss/"
    trainingMaskArraysDir = "../splits/Dataset-split-arrays/Training-Final-MSK-Arrays/"

    for dir in [trainingImgArraysDir, trainingMaskArraysDir,
                validationImgArraysDir, validationMaskArraysDir,
                testingImgArraysDir, testingMaskArraysDir]:
        try:
            shutil.rmtree(dir)
        except:
            pass
        os.makedirs(dir)

    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Testing-Final-IMG-Arrays/",
                              suffix = "testing_img_array", data = imgs_test)

    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Testing-Final-MSK-Arrays/",
                              suffix = "testing_msk_array", data = imgs_mask_test_gt)

    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Training-Final-IMG-Arrayss/",
                              suffix = "training_img_array", data = imgs_train)
    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Training-Final-MSK-Arrays/",
                              suffix = "training_msk_array", data = imgs_mask_train)

    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Validation-Final-IMG-Arrays/",
                              suffix = "validation_img_array", data = imgs_val)
    __saveDatasetArraysForNet(folderPath = "../splits/Dataset-split-arrays/Validation-Final-MSK-Arrays/",
                              suffix = "validation_msk_array", data = imgs_mask_val)


if __name__ == '__main__':
    # 12:23
    # routineMakeFinalDataset()
    saveDatasetArraysForNet()

    # BCDR -> 485
    # CBIS calc -> 1505
    # CBIS mass -> 1592
    # csaw -> 339
    # inbreast -> 118
    # total : 3.939

    # INBREAST, CBIS CALC, BCDR, CSAW
