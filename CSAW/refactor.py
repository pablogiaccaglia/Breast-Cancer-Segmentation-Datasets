import glob
import os
import shutil
import re


def _moveOriginalCSAWImages(origin_folder: str, target_folder: str, regex: str, filenameSuffix: str = None,
                            index = None):
    """
    Find original screening in origin_folder and copy it to target_folder.
    """

    if not regex:
        return

    # Define path to all png files in the folder
    path = origin_folder + "/*.png"

    # Create list with addresses of all files in the folder
    addrs = sorted(glob.glob(path))

    # Get screenings path
    if regex[0] == 'cancer.png' or regex[0] == "mammary_gland.png":
        image_list = [i for i in addrs if regex[0] in i]
        number_of_images = len(image_list)

    else:
        all_images = re.compile(regex[0])
        image_list = list(filter(all_images.match, addrs))
        number_of_images = len(image_list)

    # If no images
    if number_of_images == 0:
        try:
            all_images = re.compile(regex[1])
            image_list = list(filter(all_images.match, addrs))
        except:
            return index

    # Copy to new folder
    for image in image_list:

        if filenameSuffix and index:

            newFileName = filenameSuffix + str(index) + '.png'
            index = index + 1
            newPath = os.path.join(target_folder, newFileName)
            shutil.copy(image, newPath)
        else:

            shutil.copy(image, target_folder)

    return index

def moveCSAWImages(origin_folder: str, target_folder: str, mode, file_index: int, file_name_suffix):
    if mode == "images":
        regex = [".*_[0-9].png", ".*_[0-9][0-9].png"]

    elif mode == "masks":
        regex = ["cancer.png"]

    elif mode == "glands":
        regex = ["mammary_gland.png"]

    else:
        regex = []

    file_index_updated = _moveOriginalCSAWImages(origin_folder = origin_folder,
                                                 target_folder = target_folder,
                                                 regex = regex, filenameSuffix = file_name_suffix,
                                                 index = file_index
                                                 )

    return file_index_updated

def _routineRefactorCSAW(origin_folder, target_folder, mode, file_index, file_name_suffix):
    for dirPath, dirNames, _ in os.walk(origin_folder):

        dirNames.sort()

        for d in dirNames:
            subDirPath = os.path.join(dirPath, d)
            file_index = moveCSAWImages(origin_folder = subDirPath,
                                        target_folder = target_folder,
                                        mode = mode,
                                        file_index = file_index,
                                        file_name_suffix = file_name_suffix)

    return file_index

def refactorCSAW():
    file_index = 1
    annotatorNumber = 3
    baseOriginalTestingMSKFolder = "../CSAW/CsawS_original_folder/test_data/annotator_"
    originalTestingMSKFolder = baseOriginalTestingMSKFolder + str(annotatorNumber)

    originalTestingIMGFolder = "../CSAW/CsawS_original_folder/test_data/anonymized_dataset"

    originalTrainingMSKFolder = "../CSAW/CsawS_original_folder/anonymized_dataset"
    originalTrainingIMGFolder = "../CSAW/CsawS_original_folder/anonymized_dataset"

    targetImgsFolder = "../CSAW/intermediate/CSAW-Original-IMG"
    targetMsksFolder = "../CSAW/intermediate/CSAW-Original-MSK"
    targetMammaryGlandMsksFolder = "../CSAW/intermediate/CSAW-Original-Mammary-Gland"

    imagesFileNameSuffix = "CSAW-Image-"
    masksFileNameSuffix = "CSAW-Mask-"
    targetMammaryGlandMsksSuffix = "Mammary-gland-"

    for dir in [targetImgsFolder, targetMsksFolder, targetMammaryGlandMsksFolder]:
        shutil.rmtree(dir, ignore_errors = True)
        os.makedirs(dir)

    file_index = _routineRefactorCSAW(origin_folder = originalTestingIMGFolder,
                                      target_folder = targetImgsFolder,
                                      mode = "images",
                                      file_index = file_index,
                                      file_name_suffix = imagesFileNameSuffix)

    _routineRefactorCSAW(origin_folder = originalTrainingIMGFolder,
                         target_folder = targetImgsFolder,
                         mode = "images",
                         file_index = file_index,
                         file_name_suffix = imagesFileNameSuffix)

    file_index = 1

    file_index = _routineRefactorCSAW(origin_folder = originalTestingMSKFolder,
                                      target_folder = targetMsksFolder,
                                      mode = "masks",
                                      file_index = file_index,
                                      file_name_suffix = masksFileNameSuffix)

    _routineRefactorCSAW(origin_folder = originalTrainingMSKFolder,
                         target_folder = targetMsksFolder,
                         mode = "masks",
                         file_index = file_index,
                         file_name_suffix = masksFileNameSuffix)

    annotatorNumber = 1
    baseOriginalTestingMSKFolder = "../CSAW/CsawS_original_folder/test_data/annotator_"
    originalTestingMSKFolder = baseOriginalTestingMSKFolder + str(annotatorNumber)

    file_index = 1

    file_index = _routineRefactorCSAW(origin_folder = originalTestingMSKFolder,
                                      target_folder = targetMammaryGlandMsksFolder,
                                      mode = "glands",
                                      file_index = file_index,
                                      file_name_suffix = targetMammaryGlandMsksSuffix)

    _routineRefactorCSAW(origin_folder = originalTrainingMSKFolder,
                         target_folder = targetMammaryGlandMsksFolder,
                         mode = "glands",
                         file_index = file_index,
                         file_name_suffix = targetMammaryGlandMsksSuffix)


if __name__ == '__main__':
    refactorCSAW()
