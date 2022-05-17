import numpy as np
import cv2

from preprocessing import fullMammoPreprocess
from preprocessing import maskPreprocess
from preprocessing import getMaskPatch

path = "/Users/pablo/Desktop/nl2-project/CBIS/Dataset-split/CBIS-Validation-Final-IMG/BCDR-Validation-IMG__00002.png"
img = cv2.imread(path)
img2 = cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)
import matplotlib.pyplot as plt

ima = img[:,:,0]


print(np.equal(im, img2).all())