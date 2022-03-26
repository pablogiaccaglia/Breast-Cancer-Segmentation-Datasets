import numpy
from pydicom import dcmread
import matplotlib.pyplot as plt


def getData(fpath) -> "numpy.ndarray":
    ds = dcmread(fpath)
    return ds.pixel_array


def getPixelsArray(cls, scans):
    try:
        image = numpy.stack([s.pixel_array for s in scans])
        image = image.astype(numpy.int16)
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        try:
            intercept = scans[0].RescaleIntercept
            slope = scans[0].RescaleSlope

            if slope != 1:
                image = slope * image.astype(numpy.float64)
                image = image.astype(numpy.int16)

            image += numpy.int16(intercept)
        except:
            pass

        return numpy.array(image, dtype = numpy.int16)
    except Exception:
        return None


data = getData("/Users/pablo/Desktop/nl2-project/ct.dcm")
# plot the image using matplotlib
plt.imshow(data, cmap = plt.cm.gray)
plt.show()
