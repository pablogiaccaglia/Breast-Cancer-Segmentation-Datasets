# Breast-Cancer-Segmentation-Datasets
🩺 Curated collection of datasets for breast cancer segmentation

- 📙 [Description](#-description)
- 🗄️ [Datasets](#Datasets)
  - 🩻 [BCDR](#BCDR)
  - 🩻 [CBIS-DDSM](#CBIS-DDSM)
  - 🩻 [CSAW-S](#CSAW-S)
  - 🩻 [INbreast](#INbreast)
  - 🩻 [CDD-CESM](#CDD-CESM)
  
  
# Datasets
 ## BCDR
 First Iberian wide-ranging annotated [**BREAST CANCER DIGITAL REPOSITORY** (BCDR)](https://bcdr.eu). The BCDR is a compilation of Breast Cancer anonymized patients' 
 cases annotated by expert radiologists containing <ins>clinical data (detected anomalies, breast density, BIRADS classification, etc.), 
 lesions outlines, and image-based features computed from Craniocaudal and Mediolateral oblique mammography image views.</ins> 
 
 | Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 485  | TIF  | TIF  | 72  | PNG  | PNG |

Image     |  Mask |  Image     |  Mask
:-----------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/BCDR/BCDR-SELECTED-IMGS/patient_205_study_275_img_205_275_1_LO___PRE.png)| ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/BCDR/BCDR-SELECTED-MASKS/patient_205_study_275_img_205_275_1_LO_MASK___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/BCDR/BCDR-SELECTED-IMGS/patient_511_study_733_img_511_733_1_LCC___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/BCDR/BCDR-SELECTED-MASKS/patient_511_study_733_img_511_733_1_LCC_MASK___PRE.png)


- **[Project Website](https://www.researchgate.net/publication/258243150_BCDR_A_BREAST_CANCER_DIGITAL_REPOSITORY)**
- **[Paper](https://www.researchgate.net/publication/258243150_BCDR_A_BREAST_CANCER_DIGITAL_REPOSITORY)**
- **[Dataset](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/tree/master/BCDR)**

  
  ## INbreast
  The **[INbreast](https://pubmed.ncbi.nlm.nih.gov/22078258/)** database is a mammographic database, with images acquired at a Breast Centre, located in a Hospital de São João, 
  Breast Centre, Porto, Portugal. <ins>INbreast has a total of 115 cases (410 images) of which 90 cases are from women with both breasts 
  (4 images per case) and 25 cases are from mastectomy patients (2 images per case).</ins>
  Several types of lesions (masses, calcifications, asymmetries, and distortions) are included. Accurate contours made by specialists are also provided in XML format
  
| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 410  | DICOM  | XML  | 64  | PNG  | PNG |

Image     |  Mask |  Image     |  Mask
:-----------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/INbreast/INBREAST-SELECTED-IMGS/20588334_493155e17143edef_MG_L_CC_ANON___PRE.png)| ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/INbreast/INBREAST-SELECTED-MSKS/20588334___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/INbreast/INBREAST-SELECTED-IMGS/51049107_8c105bb715bf1c3c_MG_L_CC_ANON___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/INbreast/INBREAST-SELECTED-MSKS/51049107___PRE.png)

- **[Project Website](https://biokeanos.com/source/INBreast)**
- **[Paper](https://pubmed.ncbi.nlm.nih.gov/22078258/)**
- **[Dataset](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/tree/master/INbreast)**
  
  
  ## CSAW-S
  
  The **[CSAW-S](https://arxiv.org/pdf/2008.00807v2.pdf)** dataset is a companion subset of CSAW, a large cohort of mammography data gathered from the entire population of Stockholm 
  invited for screening between 2008 and 2015, [which is available for research (Dembrower et al., 2019)](https://zenodo.org/record/4030660#.YxSe3zBBxTU).  
  <ins>The CSAW-S subset contains mammography screenings from 172 different patients with  annotations for semantic segmentation. 
  The patients are split into a test set of 26 images from 23 patients and training/validation set containing 312 images from 150 patients.</ins> 
  
| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 338  | PNG  | PNG  | 152  | PNG  | PNG |

Image     |  Mask |  Image     |  Mask
:-----------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CSAW/CSAW-SELECTED-IMGS/CSAW-Image-122.png___PRE.png)| ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CSAW/CSAW-SELECTED-MASKS/CSAW-Mask-122.png___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CSAW/CSAW-SELECTED-IMGS/CSAW-Image-195.png___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CSAW/CSAW-SELECTED-MASKS/CSAW-Mask-195.png___PRE.png)
  
- **[Paper](https://arxiv.org/pdf/2008.00807v2.pdf)**
- **[Dataset](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/tree/master/CSAW)**

  ## CBIS-DDSM
  This [**CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM#22516629a13afa7b813e47d190f5fe9ac357446f) is an updated and standardized version of the  Digital Database for Screening Mammography (DDSM). The DDSM is a database of 2,620 scanned film mammography studies. 
  It contains normal, benign, and malignant cases with verified pathology information. The scale of the database along with ground truth validation makes the DDSM a useful tool in the development and testing of decision support systems. 
  The CBIS-DDSM collection includes a subset of the DDSM data selected and curated by a trained mammographer. The images have been decompressed and converted to DICOM format.

| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 2620  | DICOM  | DICOM  | 521  | PNG  | PNG |

Image     |  Mask |  Image     |  Mask
:-----------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CBIS/CBIS-MASS-SELECTED-IMGS/Mass-Test_P_00194_RIGHT_CC_FULL___PRE.png)| ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CBIS/CBIS-MASS-SELECTED-MASKS/Mass-Test_P_00194_RIGHT_CC_MASK_1___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CBIS/CBIS-MASS-SELECTED-IMGS/Mass-Test_P_00343_RIGHT_MLO_FULL___PRE.png) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CBIS/CBIS-MASS-SELECTED-MASKS/Mass-Test_P_00343_RIGHT_MLO_MASK___PRE.png)
  

- **[Project Website](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM)**
- **[Paper](https://www.nature.com/articles/sdata2017177)**
- **[Dataset](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/tree/master/CBIS)**
  
  
  ## CDD-CESM
  This dataset is a collection of **[2,006 high-resolution Contrast-enhanced spectral mammography (CESM) images](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8)** with annotations and medical reports. 
  CESM is done using the standard digital mammography equipment, with additional software that performs dual-energy image acquisition. 
  The images were converted from DICOM to JPEG. They have an average of 2355 x 1315 pixels.
  Full medical reports are also provided for each case (DOCX) along with manual segmentation annotation for the abnormal findings in each image (CSV file).  
  Each image with its corresponding manual annotation (breast composition, mass shape, mass margin, mass density, architectural distortion, asymmetries, calcification type, calcification distribution, mass enhancement pattern, non-mass enhancement pattern, non-mass enhancement distribution, and overall BIRADS assessment) is compiled into 1 Excel file.
  
| Size  | Original Image Format | Original Mask Format | Selected Size | Selected Image format | Selected Mask format |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 1003  | JPG  | JPG  | -  | PNG  | PNG |


Image     |  Mask |  Image     |  Mask
:-----------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CDD/CDD-SELECTED-IMGS/P177_L_DM_MLO.jpg)| ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CDD/CDD-SELECTED-MASKS/P177_L_DM_MLO_MASK.jpg) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CDD/CDD-SELECTED-IMGS/P67_R_DM_MLO.jpg) | ![](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/blob/master/CDD/CDD-SELECTED-MASKS/P67_R_DM_MLO_MASK.jpg)
  
- **[Project Website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611#109379611bcab02c187174a288dbcbf95d26179e8)**
- **[Dataset](https://github.com/pablogiaccaglia/Breast-Cancer-Segmentation-Datasets/tree/master/CDD)**
  
