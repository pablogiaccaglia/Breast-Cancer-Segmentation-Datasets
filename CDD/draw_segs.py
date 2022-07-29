from __future__ import absolute_import, division

import os
import numpy as np
import cv2
import pandas as pd
import json

from PIL import Image
from PIL import ImageDraw


def get_polygon_formatted(x_points, y_points):
    points = []
    for i in range(len(x_points)):
        points.append((x_points[i], y_points[i]))
    return points


def get_segmented_image(image, masks):
    img_mask = Image.new('L', (image.shape[1], image.shape[0]), 0)
    for mask in masks:
        if mask == '{}':
            continue
        mask = json.loads(mask)
        if mask['name'] == 'polygon':
            poly = get_polygon_formatted(mask['all_points_x'], mask['all_points_y'])
            ImageDraw.Draw(img_mask).polygon(poly, outline=1, fill=1)
    return img_mask


WRITE_PATH = '/Users/pablo/Desktop/nl2-project/CDD/real_segmentations'
ANNOTATION_CSV_FILE = '/Users/pablo/Desktop/nl2-project/CDD/Radiology_hand_drawn_segmentations_v2.csv'

df = pd.read_csv(ANNOTATION_CSV_FILE)
try:
    os.makedirs(WRITE_PATH)
except:
    print("i already exists")

imagesPath = "/Users/pablo/Desktop/nl2-project/CDD/PKG - CDD-CESM/Low energy images of CDD-CESM/"

entries = os.listdir(imagesPath)
entries.sort()

for entry in entries:
    masks = df[df['#filename'] == entry]['region_shape_attributes']
    image = cv2.imread(imagesPath + entry, cv2.cv2.IMREAD_GRAYSCALE)
    GT_mask = np.array(get_segmented_image(image, masks))

    """    if GT_mask.sum() == 0 or GT_mask.sum() > 26000:
            os.remove(imagesPath + entry)
            continue"""

    GT_mask = GT_mask*255

    maskName = entry.split('.jpg')[0] + '_MASK' + '.jpg'
    cv2.imwrite(os.path.join(WRITE_PATH, maskName), GT_mask)
