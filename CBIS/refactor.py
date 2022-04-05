from typing import Union
import numpy as np
import pydicom
import pandas as pd
import os
import shutil


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
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC/1.dcm"
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
                    pass
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

        for (curDir, dirs, files) in os.walk(top = topDirectory, topdown = False):

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
                    shutil.rmtree(curDir, ignore_errors = True)

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
                    list(zip(curDirectoryList, filesList)), columns = ["curDir", "files"]
            )
            pathToSave = os.path.join(errorDirectory, "not-empty-folders.csv")
            notEmptyDirs.to_csv(pathToSave, index = False)

    except Exception as e:
        # logger.error(f'Unable to delete_empty_folders!\n{e}')
        print(f"Unable to delete_empty_folders!\n{e}")


def moveFiles(logger, topDirPath: str, substring: str, extension: str, destinationDirectory: str) -> int:
    """
    This function recursively walks through a given directory
    (`topDirPath`) using depth-first search (bottom up), finds file names
    containing the `substr` substring and copies it to the
    target directory `destinationDirectory`.

    Parameters
    ----------
    topDirPath : {str}
        The directory to look in.
    substring : {str}
        The substring to look for, either "FULL" or "MASK".
    extension : {str}
        The extension of the file to look for. e.g. ".png".
    destinationDirectory : {str}
        The directory to copy to.

    Returns
    -------
    movedFiles : {int}
        The number of files moved.
    """

    movedFiles = 0

    try:

        # Count number of .dcm files in topDirPath.
        for currentDirectory, _, files in os.walk(topDirPath):

            files.sort()

            for f in files:

                if f.endswith(extension) and substring in f:
                    sourcePath = os.path.join(currentDirectory, f)
                    destinationPath = os.path.join(destinationDirectory, f)
                    shutil.move(sourcePath, destinationPath)

                    movedFiles += 1
                # if movedFiles == 1:
                #     break

    except Exception as e:
        # logger.error(f'Unable to moveFiles!\n{e}')
        print(f"Unable to moveFiles!\n{e}")

    return movedFiles


def updateDcmPath(logger, og_df, images_folder, masks_folder):
    """
    This function updates paths to the full mammogram scan,
    cropped image and ROI mask of each row (.dcm file) of the
    given DataFrame.
    Parameters
    ----------
    og_df : {pd.DataFrame}
        The original Pandas DataFrame that needs to be updated.
    dcm_folder : {str}
        The relative (or absolute) path to the folder that conrains
        all the .dcm files to get the path.
    Returns
    -------
    og_df: {pd.DataFrame}
        The Pandas DataFrame with all the updated .dcm paths.
    """

    try:

        # Creat new columns in og_df.
        og_df["full_path"] = np.nan
        # og_df["crop_path"] = np.nan
        og_df["mask_path"] = np.nan

        # Get list of .dcm paths.
        images_paths_list = []
        masks_paths_list = []
        for _, _, files in os.walk(images_folder):
            for f in files:
                if f.endswith(".png"):
                    images_paths_list.append(os.path.join(images_folder, f))

        for _, _, files in os.walk(masks_folder):
            for f in files:
                if f.endswith(".png"):
                    masks_paths_list.append(os.path.join(masks_folder, f))

        for row in og_df.itertuples():

            row_id = row.Index

            # Get identification details.
            patient_id = row.patient_id
            img_view = row.image_view
            lr = row.left_or_right_breast
            abnormality_id = row.abnormality_id

            # Use this list to match DF row with .dcm path.
            info_list = [patient_id, img_view, lr]

            #  crop_suffix = "CROP_" + str(abnormality_id)
            mask_suffix = "MASK_" + str(abnormality_id)

            # Get list of relevant paths to this patient.
            full_paths = [
                path
                for path in images_paths_list
                if all(info in path for info in info_list + ["FULL"])
            ]

            """ crop_paths = [
                    path
                    for path in dcm_paths_list
                    if all(info in path for info in info_list + [crop_suffix])
                ] """

            mask_paths = [
                path
                for path in masks_paths_list
                if all(info in path for info in info_list + [mask_suffix])
            ]

            # full_paths_str = ",".join(full_paths)
            # crop_paths_str = ",".join(crop_paths)
            # mask_paths_str = ",".join(mask_paths)

            # Update paths.
            if len(full_paths) > 0:
                og_df.loc[row_id, "full_path"] = full_paths
            #  if len(crop_paths) > 0:
            #      og_df.loc[row_id, "crop_path"] = crop_paths
            if len(mask_paths) > 0:
                og_df.loc[row_id, "mask_path"] = mask_paths

        del og_df["cropped_image_file_path"]
        del og_df["image_file_path"]
        del og_df["ROI_mask_file_path"]

    except Exception as e:
        # logger.error(f'Unable to updateDcmPath!\n{e}')
        print(f"Unable to get updateDcmPath!\n{e}")

    return og_df


# ----------------------------------

def refactorCBIS(logger, topDirectory: str):
    """main function for extractDicom module.
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images in the output path specified.
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    """

    # ==============================================
    # 1. Count number of .dcm files BEFORE executing
    # ==============================================
    print("start")
    before = countDcmFiles(logger = None, topDirectory = topDirectory)

    # ==========
    # 2. Execute
    # ==========

    print(before)

    # 2.1. Rename and move .dcm files.
    # --------------------------------
    for (currentDirectory, dirs, files) in os.walk(top = topDirectory, topdown = False):

        dirs.sort()
        files.sort()

        for f in files:

            # === Step 1: Rename .dcm file ===
            if f.endswith(".dcm"):

                old_name_path = os.path.join(currentDirectory, f)
                newFilename = renameDcmFiles(logger = None, dcmFilePath = old_name_path)

                if newFilename:
                    pathOfNewNameFile = os.path.join(currentDirectory, newFilename)
                    os.rename(old_name_path, pathOfNewNameFile)

                    # === Step 2: Move RENAMED .dcm file ===
                    moveDcmFileUp(logger = None,
                                  destinationDir = topDirectory, sourceDir = pathOfNewNameFile,
                                  dcmFilename = newFilename
                                  )

    # 2.2. Delete empty folders.
    # --------------------------
    deleteEmptyFolders(logger = None, topDirectory = topDirectory, errorDirectory = "/Users/pablo/Desktop/nl2-project")

    # =============================================
    # 3. Count number of .dcm files AFTER executing
    # =============================================
    after = countDcmFiles(logger = None, topDirectory = topDirectory)

    print(f"BEFORE --> Number of .dcm files: {before}")
    print(f"AFTER --> Number of .dcm files: {after}")
    print()
    print("Getting out of extractDicom.")
    print("-" * 30)

    return


def updateCSV(logger, mass_csv_path, mass_png_folder, masks_folder, output_csv_path):
    """main function for updateDcmPath module.
    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    # Read the .csv files.
    og_mass_df = pd.read_csv(mass_csv_path)

    new_cols = [col.replace(" ", "_") for col in og_mass_df.columns]
    og_mass_df.columns = new_cols

    # Update .png paths.
    updated_mass_df = updateDcmPath(
            logger = None,
            og_df = og_mass_df, images_folder = mass_png_folder, masks_folder = masks_folder
    )

    updated_mass_df.to_csv(output_csv_path, index = False)

    print("Getting out of updateDcmPath.")
    print("-" * 30)

    return


# 10223
if __name__ == '__main__':
    """
    refactorCBIS(logger = None,
                 topDirectory = "/Users/pablo/Desktop/CBIS/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM")
ds = pydicom.filereader.dcmread(
        "/Users/pablo/Desktop/CBIS/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/P_01867_LEFT_MLO_MASK_2___a.dcm")
print(ds)

ds = pydicom.filereader.dcmread(
        "/Users/pablo/Desktop/CBIS/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/P_00038_RIGHT_MLO_FULL.dcm")
print(ds)

    """
    mass_csv_path = "/Users/pablo/Desktop/nl2-project/CBIS/mass_case_description_train_set.csv"
    mass_png_folder = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Preprocessed-IMG"
    output_csv_path = "/CBIS/mass_case_description_train_set_UPDATED.csv"
    masks_folder = "/Users/pablo/Desktop/nl2-project/CBIS/CBIS-Training-Preprocessed-MSK"

    updateCSV(logger = None, mass_csv_path = mass_csv_path, mass_png_folder = mass_png_folder,
              masks_folder = masks_folder,
              output_csv_path = output_csv_path)
