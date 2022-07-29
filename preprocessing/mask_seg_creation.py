"""from pathlib import Path
import highdicom as hd
import matplotlib.pyplot as plt
from pydicom import Dataset
from pydicom import Sequence
from pydicom.sr.codedict import codes"""
from pydicom.filereader import dcmread
"""import numpy as np
from preprocessing import minMaxNormalize
from mask_handler import overlayMask"""

f = "/Volumes/Extreme SSD/CBIS-Mass-Training/Mass-Training_P_00001_LEFT_MLO_FULL.dcm"

img = dcmread(f)
print(img)

"""image_datasets = dcmread(str(f))


# MISSING FILES CBIS !!!
image_datasets.ImageOrientationPatient = [0, 0, 0, 0]
image_datasets.ImagePositionPatient = [0, 0, 0, 0]
image_datasets.FrameOfReferenceUID = hd.UID()
image_datasets.PixelSpacing = [0, 0, 0, 0]
image_datasets.SliceThickness = 1.0
image_datasets.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
image_datasets.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"

print(image_datasets)
print("__" * 10)

# Create a binary segmentation mask
mask = dcmread(
        "/Volumes/Extreme SSD/CBIS-Mass-Training/Mass-Training_P_00199_LEFT_CC_MASK_1.dcm").pixel_array

mask, _ = overlayMask(image_datasets.pixel_array, mask)
mask = mask[:,:,1]

print(mask.max())

# mask = minMaxNormalize(None, mask)

# Describe the algorithm that created the segmentation
algorithm_identification = hd.AlgorithmIdentificationSequence(
        name = 'U-Net Segmentation',
        version = 'v1.0',
        family = codes.cid7162.ArtificialIntelligence
)

# Describe the segment
description_segment_1 = hd.seg.SegmentDescription(
        segment_number = 1,
        segment_label = 'first segment',
        segmented_property_category = codes.cid7150.AnatomicalStructure,
        segmented_property_type = codes.cid7192.Breast,
        algorithm_type = hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
        algorithm_identification = algorithm_identification,
        tracking_uid = hd.UID(),
        tracking_id = 'test segmentation of computed tomography image'
)

d = []

for i in range(0, mask.max()):
    description_segment_1 = hd.seg.SegmentDescription(
            segment_number = i+1,
            segment_label = 'first segment',
            segmented_property_category = codes.cid7150.AnatomicalStructure,
            segmented_property_type = codes.cid7192.Breast,
            algorithm_type = hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification = algorithm_identification,
            tracking_uid = hd.UID(),
            tracking_id = 'test segmentation of computed tomography image'
    )

    d.append(description_segment_1)
# Create the Segmentation instance
seg_dataset = hd.seg.Segmentation(
        source_images = [image_datasets],
        pixel_array = mask,
        segmentation_type = hd.seg.enum.SegmentationTypeValues.FRACTIONAL,
        segment_descriptions = d,
        series_instance_uid = hd.UID(),
        series_number = 1,
        sop_instance_uid = hd.UID(),
        instance_number = 1,
        manufacturer = 'Manufacturer',
        manufacturer_model_name = 'Model',
        software_versions = 'v1',
        device_serial_number = 'Device XYZ'
)

print(seg_dataset)

seg_dataset.save_as("seg.dcm")
image_datasets.save_as("img.dcm")

arr = seg_dataset.pixel_array
plt.imshow(arr, cmap='gray')
plt.show()
"""