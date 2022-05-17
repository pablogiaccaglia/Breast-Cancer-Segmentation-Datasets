import os
import shutil
import numpy as np
import pandas as pd
import re
from collections import Counter
import numpy
from PIL import Image, ImageDraw
import cv2


def mergeCSVs(csvPath1: str, csvPath2: str, outputFilePath, dropDuplicates: bool = True,
              columnNames: list[str] = None) -> None:
    a = pd.read_csv(csvPath1)
    b = pd.read_csv(csvPath2)
    merged = a.merge(b, how = 'outer', sort = True)
    if dropDuplicates:
        if columnNames and len(columnNames) > 0:
            merged = merged.drop_duplicates(subset = columnNames)
        else:
            merged = merged.drop_duplicates()
    merged.to_csv(outputFilePath, index = False)


def renameTifFiles(filePath: str) -> str:
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
    filePath : {str}
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

        filePath = filePath.replace('/', '_')
        splits = filePath.split('patient')
        return 'patient' + splits[1]

    except Exception as e:
        # logger.error(f'Unable to new_name_dcm!\n{e}')
        print(f"Unable to renameTifFiles!\n{e}")


def countTifFiles(topDirectory: str) -> int:
    """
    This function recursively walks through a given directory
    (`topDirectory`) using depth-first search (bottom up) and counts the
    number of .tif files present.
    Parameters
    ----------
    topDirectory : {str}
        The directory to count.
    Returns
    -------
    count : {int}
        The number of .tif files in `path`.
    """

    count = 0

    try:

        # Count number of .dcm files in ../data/Mass/Test.
        for _, _, files in os.walk(topDirectory):
            for f in files:
                if f.endswith(".tif"):
                    count += 1

    except Exception as e:
        # logger.error(f'Unable to count_dcm!\n{e}')
        print(f"Unable to count_dcm!\n{e}")

    return count


def moveTifFileUp(destinationDir: str, sourceDir: str, dcmFilename: str) -> None:
    """
    This function move a .Tif file from its given source
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
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC_FULL.tif"
    dcmFilename : {str}
        The name of the .dcm file WITH the ".dcm" extension
        but WITHOUT its (relative or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.tif".
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
            newName2 = dcmFilename.strip(".tif") + "___a.tif"
            # This moves the file into the destination while giving the file its new name.
            shutil.move(sourceDir, os.path.join(destinationDir, newName2))

    except Exception as e:
        # logger.error(f'Unable to move_dcm_up!\n{e}')
        print(f"Unable to moveTifFileUp!\n{e}")


def deleteEmptyFolders(topDirectory: str, errorDirectory: str) -> None:
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
                    f for f in os.listdir(curDir) if (not f.startswith(".") and f.endswith('.tif'))
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


def moveFiles(topDirPath: str, substring: str, extension: str, destinationDirectory: str) -> int:
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


def updateTifPath(og_df, images_folder):
    """
    This function updates paths to the full mammogram scan,
    cropped image and ROI mask of each row (.tif file) of the
    given DataFrame.
    Parameters
    ----------
    og_df : {pd.DataFrame}
        The original Pandas DataFrame that needs to be updated.
    images_folder : {str}
        The relative (or absolute) path to the folder that contains
        all the .tif files to get the path.
    Returns
    -------
    og_df: {pd.DataFrame}
        The Pandas DataFrame with all the updated .dcm paths.
    """

    try:

        # Creat new columns in og_df.
        og_df["full_path"] = np.nan

        # Get list of .dcm paths.
        images_paths_list = []
        for _, _, files in os.walk(images_folder):
            for f in files:
                if f.endswith(".tif"):
                    images_paths_list.append(os.path.join(images_folder, f))

        for row in og_df.itertuples():

            row_id = row.Index

            views = {
                '2': 'LCC',
                '4': 'LO',
                '1': 'RCC',
                '3': 'RO'
            }

            # Get identification details.
            patient_id = 'patient_' + str(row.patient_id)
            study_id = '_study_' + str(row.study_id)
            fname = '_img_' + str(row.patient_id) + '_' + str(row.study_id) + '_1_' + views[str(row.image_view)]

            # Use this list to match DF row with .dcm path.

            # Get list of relevant paths to this patient.
            full_path = [
                path
                for path in images_paths_list
                if fname in path
            ]

            # Update paths.
            if len(full_path) > 0:
                og_df.loc[row_id, "image_filename"] = full_path[0]

    except Exception as e:
        print(f"Unable to get updateTifPath!\n{e}")

    return og_df


# ----------------------------------

def reorganizeBCDRFiles(topDirectory: str):
    """main function for extractDicom module.
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images in the output path specified.
    Parameters
    """

    # ==============================================
    # 1. Count number of .dcm files BEFORE executing
    # ==============================================
    print("start")
    before = countTifFiles(topDirectory = topDirectory)

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
            if f.endswith(".tif"):

                old_name_path = os.path.join(currentDirectory, f)
                newFilename = os.path.join(currentDirectory, f)
                newFilename = renameTifFiles(filePath = newFilename)

                if newFilename:
                    pathOfNewNameFile = os.path.join(currentDirectory, newFilename)
                    os.rename(old_name_path, pathOfNewNameFile)

                    # === Step 2: Move RENAMED .tif file ===
                    moveTifFileUp(
                            destinationDir = topDirectory, sourceDir = pathOfNewNameFile,
                            dcmFilename = newFilename
                    )

    # 2.2. Delete empty folders.
    # --------------------------
    deleteEmptyFolders(topDirectory = topDirectory, errorDirectory = "/Users/pablo/Desktop/nl2-project/BCDR")

    # =============================================
    # 3. Count number of .dcm files AFTER executing
    # =============================================
    after = countTifFiles(topDirectory = topDirectory)

    print(f"BEFORE --> Number of .tif files: {before}")
    print(f"AFTER --> Number of .tif files: {after}")
    print()
    print("Getting out of extractDicom.")
    print("-" * 30)

    return


def updateCSV(input_csv_path, img_png_folder, output_csv_path):
    # Read the .csv files.
    og_mass_df = pd.read_csv(input_csv_path)

    new_cols = [col.replace(" ", "_") for col in og_mass_df.columns]
    og_mass_df.columns = new_cols

    # Update .tif paths.
    updated_df = updateTifPath(
            og_df = og_mass_df, images_folder = img_png_folder
    )

    updated_df.to_csv(output_csv_path, index = False)

    print("Getting out of updateCSV.")
    print("-" * 30)

    return


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)


def keepOnlyImagesWithMasks(imagesDirectory: str, csvPath: str, columnName: str) -> None:
    csv = pd.read_csv(csvPath)
    filesWithMask = list(csv[columnName])
    filesWithMask = natural_sort(filesWithMask)

    allFiles = os.listdir(imagesDirectory)
    allFiless = [os.path.join(imagesDirectory, f) for f in allFiles if f.endswith('.tif')]
    allFiless = natural_sort(allFiless)

    toRemove = [f for f in allFiless if f not in filesWithMask]

    for f in toRemove:
        os.remove(f)


def createMasksAndImagesFolder(csvPath: str,
                               pathsColumnName: str,
                               xCoordsColumnName: str,
                               yCoordsColumnName: str,
                               masksDirectory: str,
                               imagesDirectory) -> None:
    csv = pd.read_csv(csvPath)

    slicedCsv = csv[[pathsColumnName, xCoordsColumnName, yCoordsColumnName]]

    filesWithMask = list(slicedCsv[pathsColumnName])
    temp = []

    # Check whether the specified path exists or not
    isExist = os.path.exists(masksDirectory)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(masksDirectory)

    # Check whether the specified path exists or not
    isExist = os.path.exists(imagesDirectory)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(imagesDirectory)

    for path in filesWithMask:
        temp.append(path.split('/')[-1])

    filesWithMask = temp

    filesWithMask = natural_sort(filesWithMask)

    unique = Counter(filesWithMask)
    items = unique.items()

    paths = [item[0] for item in items if int(item[1] > 1)]

    rowsMultiMasks = []

    for i, row in slicedCsv.iterrows():
        if row[pathsColumnName].split('/')[-1] in paths:
            rowsMultiMasks.append((row[pathsColumnName], row[xCoordsColumnName], row[yCoordsColumnName]))
            slicedCsv.drop(slicedCsv.index[i])

    for i, row in slicedCsv.iterrows():
        coordsX = row[xCoordsColumnName].split(' ')[1:]
        coordsY = row[yCoordsColumnName].split(' ')[1:]

        maskHeight, maskWidth = cv2.imread(filename = row[pathsColumnName], flags = cv2.cv2.IMREAD_GRAYSCALE).shape

        path = row[pathsColumnName].split('/')[-1]
        coordinates = [(int(x), int(y)) for x, y in zip(coordsX, coordsY)]
        img = Image.new('L', (maskWidth, maskHeight), 0)
        ImageDraw.Draw(img).polygon(coordinates, outline = 1, fill = 1)
        mask = numpy.array(img) * 255
        filename = path[:-4] + '_MASK' + '.tif'
        absPath = os.path.join(masksDirectory, filename)
        cv2.imwrite(filename = absPath, img = mask)
        shutil.copy2(row[pathsColumnName], imagesDirectory)

    for row in rowsMultiMasks:

        path = row[0]
        shutil.copy2(path, imagesDirectory)
        maskHeight, maskWidth = cv2.imread(filename = path, flags = cv2.cv2.IMREAD_GRAYSCALE).shape
        img = Image.new('L', (maskWidth, maskHeight), 0)
        rowsMultiMasks.remove(row)
        found = [r for r in rowsMultiMasks if r[0] == row[0]]
        xPointsRows = row[1]
        yPointsRows = row[2]
        coordsX = xPointsRows.split(' ')[1:]
        coordsY = yPointsRows.split(' ')[1:]
        coordinates = [(int(x), int(y)) for x, y in zip(coordsX, coordsY)]

        ImageDraw.Draw(img).polygon(coordinates, outline = 1, fill = 1)

        for f in found:
            xPointsRows = f[1]
            yPointsRows = f[2]
            coordsX = xPointsRows.split(' ')[1:]
            coordsY = yPointsRows.split(' ')[1:]
            coordinates = [(int(x), int(y)) for x, y in zip(coordsX, coordsY)]
            ImageDraw.Draw(img).polygon(coordinates, outline = 1, fill = 1)

        mask = numpy.array(img) * 255

        path = path.split('/')[-1]
        filename = path[:-4] + '_MASK' + '.tif'
        absPath = os.path.join(masksDirectory, filename)
        cv2.imwrite(filename = absPath, img = mask)


def refactorBCDR():
    bcdrDatasetD01Path = "/Users/pablo/Downloads/BCDR/BCDR-D01_dataset"
    bcdrDatasetD02Path = "/Users/pablo/Downloads/BCDR/BCDR-D02_dataset"

    csvPathD02 = '../BCDR/bcdr_d02_outlines.csv'
    outputCsvPathD02 = '../BCDR/bcdr_d02_outlines_UPDATED.csv'

    csvPathD01 = '../BCDR/bcdr_d01_outlines.csv'
    outputCsvPath02 = '../BCDR/bcdr_d01_outlines_UPDATED.csv'

    outputMergedCSVPath = '../BCDR/ouput.csv'

    bcdrMasksPath = '/Users/pablo/Desktop/nl2-project/BCDR/BCDR-Masks'
    bcdrImagesPath = '/Users/pablo/Desktop/nl2-project/BCDR/BCDR-Images'

    xCoordsColumnName = 'lw_x_points'
    yCoordsColumnName = 'lw_y_points'

    reorganizeBCDRFiles(topDirectory = bcdrDatasetD02Path)
    reorganizeBCDRFiles(topDirectory = bcdrDatasetD01Path)

    updateCSV(input_csv_path = csvPathD02, img_png_folder = bcdrDatasetD02Path,
              output_csv_path = outputCsvPathD02)

    updateCSV(input_csv_path = csvPathD01, img_png_folder = bcdrDatasetD01Path,
              output_csv_path = outputCsvPath02)

    """    keepOnlyImagesWithMasks(imagesDirectory = "/Users/pablo/Downloads/BCDR/BCDR-D01_dataset",
                                csvPath = "/Users/pablo/Desktop/nl2-project/BCDR/bcdr_d01_outlines_UPDATED.csv",
                                columnName = "image_filename")"""

    r = pd.read_csv(csvPathD01)
    columnNames = list(r.columns)
    columnNames.remove('image_filename')

    mergeCSVs(csvPath1 = csvPathD01,
              csvPath2 = csvPathD02,
              outputFilePath = outputMergedCSVPath,
              dropDuplicates = True,
              columnNames = columnNames)

    createMasksAndImagesFolder(csvPath = outputMergedCSVPath,
                               pathsColumnName = 'image_filename',
                               xCoordsColumnName = xCoordsColumnName,
                               yCoordsColumnName = yCoordsColumnName,
                               masksDirectory = bcdrMasksPath,
                               imagesDirectory = bcdrImagesPath)


if __name__ == '__main__':
    refactorBCDR()
