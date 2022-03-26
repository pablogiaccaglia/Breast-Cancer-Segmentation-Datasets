import os
import shutil
from typing import Union
import numpy as np

import pydicom
import pandas as pd


def renameDcmFiles(logger, dcmFilePath: str) -> Union[str, bool]:

    """
    This function takes the absolute path of a .dcm file
    and renames it according to the convention below:
    1. Full mammograms:
        - Mass-Training_P_00001_LEFT_CC_FULL.dcm
    2. Cropped image:
        - Mass-Training_P_00001_LEFT_CC_CROP_1.dcm
        - Mass-Training_P_00001_LEFT_CC_CROP_2.dcm
        - ...
    3. Mask image:
        - Mass-Training_P_00001_LEFT_CC_MASK_1.dcm
        - Mass-Training_P_00001_LEFT_CC_MASK_2.dcm
        - ...
    Parameters
    ----------
    dcmFilePath : {str}
        The relative (or absolute) path of the .dcm file
        to rename, including the .dcm filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC/1-1.dcm"
    Returns
    -------
    newFilename : {str}
        The new name that the .dcm file should have
        WITH the ".dcm" extention WITHOUT its relative
        (or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    False : {boolean}
        False is returned if the new name of the .dcm
        file cannot be determined.
    """

    try:
        # Read dicom.
        ds = pydicom.dcmread(dcmFilePath)

        # Get information.
        patientID = ds.PatientID
        patientID = patientID.replace(".dcm", "")

        try:
            # If ds contains SeriesDescription attribute...
            imageType = ds.SeriesDescription

            # === FULL ===
            if "full" in imageType:
                newFilename = patientID + "_FULL" + ".dcm"
                print(f"FULL --- {newFilename}")
                return newFilename

            # === CROP ===
            elif "crop" in imageType:

                # Double check if suffix is integer.
                suffix = patientID.split("_")[-1]

                if suffix.isdigit():
                    newPatientID = patientID.split("_" + suffix)[0]
                    newFilename = newPatientID + "_CROP" + "_" + suffix + ".dcm"
                    print(f"CROP --- {newFilename}")
                    return newFilename

                elif not suffix.isdigit():
                    print(f"CROP ERROR, {patientID}")
                    return False

            # === MASK ===
            elif "mask" in imageType:

                # Double check if suffix is integer.
                suffix = patientID.split("_")[-1]

                if suffix.isdigit():
                    newPatientID = patientID.split("_" + suffix)[0]
                    newFilename = newPatientID + "_MASK" + "_" + suffix + ".dcm"
                    print(f"MASK --- {newFilename}")
                    return newFilename

                elif not suffix.isdigit():
                    print(f"MASK ERROR, {patientID}")
                    return False

        except:
            # If ds does not contain SeriesDescription...
            # === FULL ===
            if "full" in dcmFilePath:
                newFilename = patientID + "_FULL" + ".dcm"
                return newFilename

            else:
                # Read the image to decide if its a mask or crop.
                # MASK only has pixel values {0, 1}
                arr = ds.pixel_array
                unique = np.unique(arr).tolist()

                if len(unique) != 2:

                    # === CROP ===
                    # Double check if suffix is integer.
                    suffix = patientID.split("_")[-1]

                    if suffix.isdigit():
                        newPatientID = patientID.split("_" + suffix)[0]
                        newFilename = newPatientID + "_CROP" + "_" + suffix + ".dcm"
                        print(f"CROP --- {newFilename}")
                        return newFilename

                    elif not suffix.isdigit():
                        print(f"CROP ERROR, {patientID}")
                        return False

                else:

                    # === MASK ===
                    # Double check if suffix is integer.
                    suffix = patientID.split("_")[-1]

                    if suffix.isdigit():
                        newPatientID = patientID.split("_" + suffix)[0]
                        newFilename = newPatientID + "_MASK" + "_" + suffix + ".dcm"
                        print(f"MASK --- {newFilename}")
                        return newFilename

                    elif not suffix.isdigit():
                        print(f"MASK ERROR, {patientID}")
                        return False

    except Exception as e:
        # logger.error(f'Unable to new_name_dcm!\n{e}')
        print(f"Unable to new_name_dcm!\n{e}")

def countDcmFiles(logger, topDirectory: str) -> int:

    """
    This function recursively walks through a given directory
    (`topDirectory`) using depth-first search (bottom up) and counts the
    number of .dcm files present.
    Parameters
    ----------
    path : {str}
        The directory to count.
    Returns
    -------
    count : {int}
        The number of .dcm files in `path`.
    """

    count = 0

    try:

        # Count number of .dcm files in ../data/Mass/Test.
        for _, _, files in os.walk(topDirectory):
            for f in files:
                if f.endswith(".dcm"):
                    count += 1

    except Exception as e:
        # logger.error(f'Unable to count_dcm!\n{e}')
        print(f"Unable to count_dcm!\n{e}")

    return count

def moveDcmFileUp(logger, destinationDir: str, sourceDir: str, dcmFilename: str) -> None:

    """
    This function move a .dcm file from its given source
    directory into the given destination directory. It also
    handles conflicting filenames by adding "___a" to the
    end of a filename if the filename already exists in the
    destination directory.
    Parameters
    ----------
    destinationDir : {str}
        The relative (or absolute) path of the folder that
        the .dcm file needs to be moved to.
    sourceDir : {str}
        The relative (or absolute) path where the .dcm file
        needs to be moved from, including the filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    dcmFilename : {str}
        The name of the .dcm file WITH the ".dcm" extension
        but WITHOUT its (relative or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm".
    Returns
    -------
    None
    """

    try:
        dest_dir_with_new_name = os.path.join(destinationDir, dcmFilename)

        # If the destination path does not exist yet...
        if not os.path.exists(dest_dir_with_new_name):
            shutil.move(sourceDir, destinationDir)

        # If the destination path already exists...
        elif os.path.exists(dest_dir_with_new_name):
            # Add "_a" to the end of `new_name` generated above.
            newName2 = dcmFilename.strip(".dcm") + "___a.dcm"
            # This moves the file into the destination while giving the file its new name.
            shutil.move(sourceDir, os.path.join(destinationDir, newName2))

    except Exception as e:
        # logger.error(f'Unable to move_dcm_up!\n{e}')
        print(f"Unable to move_dcm_up!\n{e}")

def deleteEmptyFolders(logger, topDirectory: str, errorDirectory: str) -> None:

    """
    This function recursively walks through a given directory
    (`topDirectory`) using depth-first search (bottom up) and deletes
    any directory that is empty (ignoring hidden files).
    If there are directories that are not empty (except hidden
    files), it will save the absolute directory in a Pandas
    dataframe and export it as a `not-empty-folders.csv` to
    `error_dir`.
    Parameters
    ----------
    topDirectory : {str}
        The directory to iterate through.
    errorDirectory : {str}
        The directory to save the `not-empty-folders.csv` to.
    Returns
    -------
    None
    """

    try:
        curDirectoryList = []
        filesList = []

        for (curDir, dirs, files) in os.walk(top=topDirectory, topdown=False):

            if curDir != str(topDirectory):

                dirs.sort()
                files.sort()

                print(f"WE ARE AT: {curDir}")
                print("=" * 10)

                print("List dir:")

                directories_list = [
                    f for f in os.listdir(curDir) if not f.startswith(".")
                ]
                print(directories_list)

                if len(directories_list) == 0:
                    print("DELETE")
                    shutil.rmtree(curDir, ignore_errors=True)

                elif len(directories_list) > 0:
                    print("DON'T DELETE")
                    curDirectoryList.append(curDir)
                    filesList.append(directories_list)

                print()
                print("Moving one folder up...")
                print("-" * 40)
                print()

        if len(curDirectoryList) > 0:
            notEmptyDirs = pd.DataFrame(
                list(zip(curDirectoryList, filesList)), columns=["curDir", "files"]
            )
            pathToSave = os.path.join(errorDirectory, "not-empty-folders.csv")
            notEmptyDirs.to_csv(pathToSave, index=False)

    except Exception as e:
        # logger.error(f'Unable to delete_empty_folders!\n{e}')
        print(f"Unable to delete_empty_folders!\n{e}")