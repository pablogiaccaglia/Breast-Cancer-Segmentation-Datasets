             INbreast Release 1.0
Breast Research Group, INESC Porto, Portugal
http://medicalresearch.inescporto.pt/breastresearch/
         medicalresearch@inescporto.pt 

- The complete set of dicom images can be found in the folder 'AllDICOMs'.
The filename includes an anonymized patient ID. For instance, in the filename ‘20586908_6c613a14b80a8591_MG_R_CC_ANON’, ‘6c613a14b80a8591’ is the patient ID that can be used to aggregate DICOMs in cases and to link dicoms with the medical report.
A small matlab script is provided as example to process the images.


- The manual annotation is provided under two different formats: '.roi' files and '.xml' files.
'.roi' files were created with the Osirix software (http://www.osirix-viewer.com/).
We strongly recommend you to install the Osirix, open one of the images and import the corresponding '.roi' file.
The '.xml' files were also created with the Osirix software, now by exporting the annotations.

- The INbreast.xls file contains a summary of the database, including the BIRADS classification.
The date field corresponds to the year and semester (01 or 02).

- The INbreast.csv is a subset of the INbreast.xls file, for easier integration with Matlab.

- The INbreast.pdf file is the associated publication in Academic Radiology.

- The folder 'MedicalReports' contains the associated medical reports.
The filename is the anonymized patient ID. The date, when present, corresponds to the year and semester (01 or 02).

- The folder 'PectoralMuscle' contains the manual annotation of the pectoral muscle boundary.



