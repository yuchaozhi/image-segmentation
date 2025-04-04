# Image Segmentation and Automated Classification of Alzheimer's Disease
This repository provides a reproducible machine learning framework for the automated classification of Alzheimer's Disease (AD) using image segmentation techniques. It leverages both Random Forest (RF) and Support Vector Machine (SVM) algorithms to perform the classification based on processed imaging data.

## Project Background
Alzheimer's Disease is a prevalent neurodegenerative disorder affecting memory and cognitive functions. Early diagnosis and effective classification are crucial for timely treatment. This project uses advanced image preprocessing methods, such as Contrast Limited Adaptive Histogram Equalization (CLAHE), to enhance imaging data before applying machine learning algorithms for accurate AD detection.

## Data Preparation
Detailed data is available from the following sources:
- [OASIS Neuroimaging Dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- [Alzheimer's MRI 4-class Dataset](https://www.kaggle.com/datasets/marcopinamonti/alzheimer-mri-4-classes-dataset)
Download the datasets from the provided links and place the files in the appropriate directory. The data loading and preprocessing procedures are implemented in the dataset_adni.py file.

## Repository Structure
image-segmentation/

├── CLAHE.py               # Implements CLAHE for image contrast enhancement.

├── RF.py                  # Contains code for classification using the Random Forest algorithm.

├── SVM.py                 # Contains code for classification using the Support Vector Machine algorithm.

├── dataset_adni.py        # Handles data loading and preprocessing.

├── entropy.py             # Calculates image entropy for feature extraction.

├── exception_detection/   # Contains modules for exception detection and handling.

└── t_test.py              # Performs statistical T-tests to evaluate the results.


### Module Descriptions

- **CLAHE.py**  
  Enhances image contrast using the Contrast Limited Adaptive Histogram Equalization (CLAHE) technique.

- **RF.py**  
  Implements the Random Forest algorithm for classifying images, ideal for modeling non-linear data.

- **SVM.py**  
  Implements the Support Vector Machine algorithm for classifying images, especially effective with high-dimensional data.

- **dataset_adni.py**  
  Provides functions for loading, preprocessing, and splitting the dataset. Adjust the data paths as needed.

- **entropy.py**  
  Computes the entropy of images to serve as a feature extraction method.

- **exception_detection/**  
  Contains scripts and modules for detecting and managing any exceptions during the data processing or training phases.

- **t_test.py**  
  Conducts T-tests to statistically evaluate the significance of the experimental results.

## Environment Requirements

This project is developed using Python 3.x. You will need the following libraries:

- numpy
- scipy
- scikit-learn
- opencv-python
- matplotlib

Install the dependencies with pip:

```bash
pip install numpy scipy scikit-learn opencv-python matplotlib
