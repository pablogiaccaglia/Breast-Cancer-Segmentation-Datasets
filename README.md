# Breast-Cancer-Segmentation-Datasets
🩺 Curated collection of datasets for breast cancer segmentation

- 📙 [Description](#-description)
- 🗄️ [Datasets](Datasets)
  - 🩻 [BCDR](#-BCDR)
  - 🩻 [CBIS-DDSM](#-CBIS)
  - 🩻 [CSAW-S](#-CSAW-S)
  - 🩻 [INbreast](#-INbreast)
  - 🩻 [CDD-CESM](#-CDD-CESM)
  
  
 
 # BCDR
 First Iberian wide-ranging annotated [**BREAST CANCER DIGITAL REPOSITORY** (BCDR)](https://bcdr.eu). The BCDR is a compilation of Breast Cancer anonymized patients' 
 cases annotated by expert radiologists containing <ins>clinical data (detected anomalies, breast density, BIRADS classification, etc.), 
 lesions outlines, and image-based features computed from Craniocaudal and Mediolateral oblique mammography image views.</ins> 
 
 | Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 485  | TIF  | TIF  | 72  | PNG  | PNG |
  
  # INbreast
  The **[INbreast](https://pubmed.ncbi.nlm.nih.gov/22078258/)** database is a mammographic database, with images acquired at a Breast Centre, located in a Hospital de São João, 
  Breast Centre, Porto, Portugal. <ins>INbreast has a total of 115 cases (410 images) of which 90 cases are from women with both breasts 
  (4 images per case) and 25 cases are from mastectomy patients (2 images per case).</ins>
  Several types of lesions (masses, calcifications, asymmetries, and distortions) are included. Accurate contours made by specialists are also provided in XML format
  
| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 410  | DICOM  | XML  | 64  | PNG  | PNG |
  
  # CSAW-S
  
  The **[CSAW-S](https://arxiv.org/pdf/2008.00807v2.pdf)** dataset is a companion subset of CSAW, a large cohort of mammography data gathered from the entire population of Stockholm 
  invited for screening between 2008 and 2015, [which is available for research (Dembrower et al., 2019)](https://zenodo.org/record/4030660#.YxSe3zBBxTU).  
  <ins>The CSAW-S subset contains mammography screenings from 172 different patients with  annotations for semantic segmentation. 
  The patients are split into a test set of 26 images from 23 patients and training/validation set containing 312 images from 150 patients.</ins> 
  
| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 338  | PNG  | PNG  | 152  | PNG  | PNG |

  # CBIS-DDSM
  This [**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#22516629a13afa7b813e47d190f5fe9ac357446f) is an updated and standardized version of the  Digital Database for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. 
  It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems. 
  The CBIS-DDSM collection includes a subset of the DDSM data selected and curated by a trained mammographer. The images have been decompressed and converted to DICOM format.

| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 2620  | DICOM  | DICOM  | 521  | PNG  | PNG |
  
  # CDD - CESM
  This dataset is a collection of **[2,006 high-resolution Contrast-enhanced spectral mammography (CESM) images](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8)** with annotations and medical reports. 
  CESM is done using the standard digital mammography equipment, with additional software that performs dual-energy image acquisition. 
  The images were converted from DICOM to JPEG. They have an average of 2355 x 1315 pixels.
  Full medical reports are also provided for each case (DOCX) along with manual segmentation annotation for the abnormal findings in each image (CSV file).  
  Each image with its corresponding manual annotation (breast composition, mass shape, mass margin, mass density, architectural distortion, asymmetries, calcification type, calcification distribution, mass enhancement pattern, non-mass enhancement pattern, non-mass enhancement distribution, and overall BIRADS assessment) is compiled into 1 Excel file.
  
  | Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 2620  | JPG  | CSV  | 521  | PNG  | PNG |
  
  
