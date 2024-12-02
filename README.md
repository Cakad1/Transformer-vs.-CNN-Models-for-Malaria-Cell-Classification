# Transformer-vs.-CNN-Models-for-Malaria-Cell-Classification
Performance Comparison of Transformer and CNN Models for Parasitized and Uninfected Cell Classification in Malaria Detection

Caleb Kadiri
Morgan State University
Email: Cakad1@morgan.edu

## Abstract
This study investigates the performance of Transformer-based models (ViT, DeiT) and Convolutional Neural Networks (CNNs) (Simple CNN, VGG16, Xception, InceptionV3, MobileNetV2, DenseNet121) and Ensemble models for the classification of parasitized and uninfected cells in malaria detection. The models were evaluated using various performance metrics, including accuracy, precision, recall, F1 score, and confusion matrix. Results demonstrate the potential of Transformer models in improving automated malaria screening systems.

## Keywords
Malaria detection, Transformer models, ViT, DeiT, CNN models, classification, F1 score, confusion matrix.

## Table of Contents
1.  [Abstract](#Abstract)
2.  [Keywords](#keywords)
3. [Introduction](#Introduction)
4. [Related Work](#Related-Work)
5. [Literature Review](#Literature-Review)
6. [Methodology](#methodology)
7. [Results](#results)
8. [Discussion](#Discussion)
9. [Conclusion](#conclusion)
10. [Recommendations](#recommendations)

## Introduction
**Background**
Malaria remains a deadly infectious disease worldwide, transmitted through infected female Anopheles mosquitoes. Early and accurate detection is essential for effective treatment and prevention, traditionally conducted through manual examination of blood smear images.

**Problem Statement**
Manual diagnosis of malaria is time-consuming and prone to human error, highlighting the need for automated systems to enhance diagnostic accuracy and efficiency.
## Objectives

This research aims to compare the performance of CNN and Transformer-based models in malaria detection, targeting over 95% classification accuracy.

## Related Work
Deep learning models have been extensively used for malaria detection. CNN-based models have demonstrated great success in image classification, while recent studies suggest the potential of Transformer-based models for improving performance in this field.

## Literature Review
**CNN Models for Malaria Detection**
CNNs like VGG16, Xception, and EfficientNet have shown notable success in classifying parasitized and uninfected cells, achieving accuracies up to 97%.

**Transformer Models for Malaria Detection**
Vision Transformers (ViTs) and Data-efficient Image Transformers (DeiTs) have emerged as powerful tools for medical image analysis, capturing long-range dependencies and context effectively.

**Hybrid and Ensemble Approaches**
Ensemble learning and hybrid models, combining CNNs and Transformers, have shown potential in capturing both local and global features for more robust classification.

## Methodology

**Dataset**

The dataset comprised 27,560 blood smear images, evenly distributed between two classes:
* Parasitized: 13,780 images
* Uninfected: 13,780 images

**Data Preprocessing**

* All images were resized to 224x224 pixels to standardize input dimensions.
* Image augmentation techniques (rotation, zooming, horizontal flipping, etc.) were applied to improve generalization and reduce overfitting.
* Images were normalized by rescaling the pixel values to a range of [0,1].

**Model Development and Training **

* CNN Models
* Simple CNN
A custom CNN architecture was developed using TensorFlow/Keras, consisting of convolutional and max-pooling layers for feature extraction.

* Transfer Learning Models
Pre-trained models (e.g., Xception, DenseNet121, VGG16) were fine-tuned by adding a custom classification head while freezing base layers.

* Transformer Models
* Vision Transformer (ViT)
The ViT model was loaded using the timm.create_model library, with a custom classification head for binary classification.

* Data-efficient Image Transformer (DeiT)
Similarly, the DeiT model was adapted by adding a custom classification layer suitable for malaria detection.

**Training Procedure**

The models were trained using the Adam optimizer, early stopping, and learning rate scheduling to prevent overfitting and ensure convergence.

## Results

**Model Evaluation**

Each model was evaluated based on accuracy, precision, recall, F1 score, and confusion matrix to gauge performance in classifying parasitized and uninfected cells.

**Training and Validation Performance**

**Training Accuracy and Loss**
The training accuracy and loss for each model were monitored, and early stopping was applied to maintain optimal performance.

**Validation Accuracy and Loss**
Validation accuracy and loss metrics were plotted to track model generalization capabilities

**Confusion Matrix Analysis**
Confusion matrices for each model, including True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN), were analyzed.

**Accuracy**

The ViT and DeiT models both achieved an accuracy of 0.96, while the Simple CNN and other CNN models demonstrated competitive accuracy levels.

**Precision**
The DeiT model achieved a precision of 0.98 for uninfected cells, while several CNN models scored similarly for parasitized cells.

**Recall**
The ViT model led with a recall of 0.97 for parasitized cells, while the DeiT model followed with a recall of 0.95.

**F1 Score**
F1 scores for the DeiT and ViT models were 0.97 and 0.96, respectively, demonstrating their strong balance between precision and recall.
**ROC Curve and AUC**

The Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) were plotted to evaluate each model's discriminatory power.

## Discussion
** Performance Comparison

The Transformer models (ViT and DeiT) showed superior performance in terms of accuracy and F1 score, even when trained on only 15% of the dataset compared to the CNN models.

**Performance of Ensemble Methods**
Ensemble models, which combined multiple CNNs and Transformers, demonstrated improved generalization and classification accuracy over individual models.

## Conclusion

The Transformer models generally outperformed CNN models across various metrics, while ensemble methods provided enhanced performance in most cases.

## Recommendations

Future research should focus on further improving model accuracy, interpretability, and the use of larger datasets to support malaria diagnosis in a wider range of settings.
