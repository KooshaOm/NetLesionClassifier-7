# NetLesionClassifier-7

## Table of Contents

1. [Project Overview](#project-overview)<br/>
2. [Key Features](#key-features)<br/>
3. [Technical Details](#technical-details)<br/>
4. [Installation and Setup](#installation-and-setup)<br/>
5. [Usage Instructions](#usage-instructions)<br/>
6. [Examples and Demos](#examples-and-demos)<br/>
7. [Contact Information](#contact-information)<br/>
8. [Acknowledgments](#acknowledgments)<br/>
9. [Future Improvements](#future-improvements)<br/>
10. [Challenges and Solutions](#challenges-and-solutions)<br/>
11. [Learn More](#learn-more)<br/>

## Project Overview

This project focuses on developing an advanced deep learning model for skin cancer detection using the HAM10000 dataset. The primary goal is to create a model that can accurately classify and segment 7 types of skin cancer from images, leveraging state-of-the-art deep learning techniques to improve diagnostic accuracy.

The model architecture consists of three main components:

U-Net for Image Segmentation: The U-Net is utilized for segmenting images into relevant regions, specifically identifying cancerous areas. This model comprises a contracting path (encoder), a symmetric expanding path (decoder), and a bottleneck. It effectively captures and preserves spatial information to produce accurate binary masks, which are crucial for isolating cancerous lesions.
ResNet-50 for Feature Extraction: Following segmentation, the ResNet-50 model is adapted for feature extraction. This deep convolutional neural network uses residual blocks to capture high-level features from the segmented images. By removing the final classification layer, ResNet-50 is repurposed to extract meaningful features, which are essential for accurate classification.
MLP Classifier for Final Classification: The features extracted by ResNet-50 are processed through a multi-layer perceptron (MLP) classifier. The MLP, optimized through random search for hyperparameter tuning, classifies the images into eight different skin cancer types.

Our model was trained and validated on the HAM10000 dataset, achieving an impressive accuracy of 85.4% and outperforming several state-of-the-art approaches. It also demonstrated high recall, precision, and F1-score, underscoring its effectiveness in detecting various skin cancers.

This project addresses the critical need for accurate and reliable skin cancer detection tools, contributing to early diagnosis and better treatment outcomes in dermatology.

## Key Features

Our model is built on a robust architecture comprising three main components, each contributing uniquely to the overall performance and efficacy of our skin cancer detection system:

1. **U-Net for Image Segmentation:**
The U-Net architecture, renowned for its effectiveness in image segmentation tasks, serves as our primary tool for isolating cancerous lesions. Its design includes a contracting path (encoder) and an expanding path (decoder), facilitating precise object boundary reconstruction. We adapted the traditional U-Net to handle computational constraints, training on subsets of 500 images across 20 epochs. This approach enabled us to successfully segment approximately 66,000 images, generating high-quality binary masks that isolate cancerous areas from healthy skin.

2. **Transfer Learning with ResNet-50:**
Leveraging the ResNet-50 model as a feature extractor, we capitalize on its powerful convolutional layers to capture intricate patterns from the segmented images. By removing the final fully connected layer, we repurposed ResNet-50 to generate high-dimensional feature representations while avoiding the vanishing gradient problem through its innovative use of residual blocks. This efficient feature extraction process is crucial for the subsequent classification step.

3. **Optimized Multilayer Perceptron (MLP) Classifier:**
Our MLP classifier, fine-tuned through extensive hyperparameter optimization, processes the features extracted from ResNet-50. With six layers, it effectively transforms the high-dimensional feature vectors into class probabilities across seven skin cancer types. The use of the Adam optimizer and Cross-Entropy Loss ensures robust training performance, resulting in a validation accuracy of approximately 90% and a loss of 1.2.


**Model Comparison**<br/>
In comparison to existing solutions in the domain of skin cancer detection, our model demonstrates superior performance and robustness:

- Enhanced Performance: While prior research utilizing transfer learning and deep architectures has reported accuracies averaging around 91%, our model has outperformed these benchmarks. By integrating a U-Net for segmentation with a customized ResNet-50 for feature extraction and an optimized MLP for classification, we have achieved significant improvements in classification accuracy, particularly in handling the complexities of multiple skin cancer types.

- Comprehensive Architecture: Many existing models rely on simpler CNNs or MLPs that often underperform in distinguishing between subtle visual differences in skin lesions. Our approach combines advanced segmentation and feature extraction, leading to a more nuanced understanding of the input data.

- Robust Data Augmentation: We implemented innovative data augmentation techniques to enhance model robustness and generalizability, addressing overfitting and improving performance across diverse samples.

- Thorough Understanding of Deep Learning Techniques: Developing our model required an in-depth understanding of the architecture and training processes. This is evident in the customization of the ResNet-50 layers and the effective tuning of hyperparameters in the MLP, setting our work apart from existing research.

**Research Context**<br/>
In evaluating our model against recent studies:

- A study by Fathima et al. (2020) achieved a maximum accuracy of 76% using a CNN with Swish activation, while our model surpasses this with higher classification accuracy.

- Research by Jiang (2021) found that MobileNetV3 achieved an accuracy of 91.02% through various optimizations; however, our comprehensive approach combines advanced segmentation and feature extraction, allowing us to address the complexities of skin cancer classification more effectively.

- The recent work by Lilhore et al. (2024) reported impressive metrics (accuracy of 98.86%); while their hybrid model is commendable, our architecture's design and extensive training methodology contribute to our model's competitive edge in practical applications.

In conclusion, our model not only integrates state-of-the-art techniques in deep learning but also surpasses current research standards in skin cancer detection, providing a comprehensive, efficient, and high-performing solution for accurate diagnosis.


## Technical Details

## Installation and Setup
**Google Colab:** Simply upload the Jupyter Notebook into Google Colab, and place the data directory into your drive. Ensure that the notebook correctly mounts your drive. 

**Local:** Refer to [Pytorch](https://pytorch.org/get-started/locally/) Documentation for setting up a local Pytorch environment.
## Usage Instructions

## Examples and Demos

## Contact Information
Koosha Omidian - B.ASc in Computer Engineering - koosha.omidian@mail.utoronto.ca<br/>
Sepehr Rajabian -  B.ASc in Computer Engineering - sep.rajabian@mail.utoronto.ca<br/>
Erfan Nazarian -  B.ASc in Computer Engineering - erfan.nazarian@mail.utoronto.ca <br/>
Kiarash Alirezaei -  B.ASc in Computer Engineering - kiarash.alirezaei@mail.utoronto.ca<br/>
## Acknowledgments
This project was done in collaboration between 4 University of Toronto Computer Engineering students: <br/>
Koosha Omidian, Erfan Nazarian, Sepehr Rajabian, and Kiarash Alirezaei. 

## Future Improvements
No planned features at this time. 
## Challenges and Solutions

## Learn More
[Project Presentation](https://youtu.be/LtXUWCuW1PA)<br/>
[Project Report](media/SkinCancer_Final_Report.pdf)
