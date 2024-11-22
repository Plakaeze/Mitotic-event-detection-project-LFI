# Version 1.0.0
## Introduction
Welcome to the mitotic event detection project for the lens-free imaging image dataset. To request for the dataset, please send an email to manorost.panason@kuleuven.be because the dataset is not yet publicly available. This project contain 2 main code files for the features extraction storing in the "Core" folder. Code for the extracted features inspection and classification test is also included in that folder.

## CORE function
The code for POC is core.py and the code for features extraction is core_rerun.py which contains subprocess are:
- image preprocessing for cell detection in the intensity image
- Events pathway generation by linking the cell in each image frame
- Path similarlity calculation to eliminate the path that have similar position more than 40%
- Features extraction from area of interest
- Features extraction from mother cell and daughter cell in telophase frame
- Features extraction from cell moving.
