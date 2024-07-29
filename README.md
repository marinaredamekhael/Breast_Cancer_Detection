# Breast Cancer Detection Using ResNet-50
![image](https://www.frontiersin.org/files/Articles/572671/fdgth-02-572671-HTML/image_m/fdgth-02-572671-g001.jpg)
## Project Overview

This project aims to enhance breast cancer detection in histopathological images by leveraging the ResNet-50 convolutional neural network (CNN). The primary focus is on accurately identifying invasive ductal carcinoma (IDC) regions within whole slide images (WSIs).

## Key Features

- **Model Architecture:** Utilized ResNet-50, a deep residual network, to improve detection accuracy.
- **Accuracy:** Achieved a validation accuracy of 97%.
- **Data Augmentation:** Employed various augmentation techniques to enhance model robustness.
- **Optimization:** Utilized RMSprop optimizer and binary cross-entropy loss.

## Theoretical Model Explanation

### Data Preprocessing

The preprocessing pipeline includes resizing images to a standard size and applying contrast enhancement techniques. Data augmentation techniques such as rotation, width and height shift, shear, zoom, and horizontal flip are used to increase the variability of the training data, making the model more robust.

### Model Architecture

#### ResNet-50

ResNet-50 is a deep residual network consisting of 50 layers. It is designed to facilitate the training of very deep networks by introducing residual connections, or skip connections, which allow the network to learn identity mappings.

- **Residual Connections:** These connections help in mitigating the vanishing gradient problem and enable the construction of very deep networks. They allow gradients to flow through the network directly, facilitating easier training.
- **Convolutional Layers:** These layers apply convolution operations to the input image, extracting features such as edges, textures, and patterns.
- **Batch Normalization:** This technique normalizes the output of each convolutional layer, accelerating the training process and improving the model's performance.
- **Activation Functions:** The Exponential Linear Unit (ELU) activation function introduces non-linearity into the model, allowing it to learn complex patterns.
  ![image](https://i.pinimg.com/originals/d3/fd/09/d3fd09f011583932b832ea64f78233af.png)

### Training and Optimization

#### RMSprop Optimizer

RMSprop is an adaptive learning rate optimization algorithm designed to maintain a per-parameter learning rate that improves the convergence rate of the model. It helps in dealing with the varying magnitudes of parameter updates.

#### Binary Cross-Entropy Loss

This loss function is used for binary classification problems, such as distinguishing between IDC-positive and IDC-negative regions. It measures the performance of a classification model whose output is a probability value between 0 and 1.

#### Early Stopping and Learning Rate Reduction

- **Early Stopping:** This technique stops the training process when the model's performance on a validation set stops improving, preventing overfitting.
- **Learning Rate Reduction:** This method reduces the learning rate when the validation performance plateaus, allowing the model to fine-tune its parameters more effectively.

## Evaluation

The model is evaluated using several metrics to ensure its reliability and accuracy:

- **Accuracy:** Measures the overall correctness of the model.
- **Precision:** Indicates the accuracy of positive predictions.
- **Recall:** Reflects the model's ability to capture all positive instances.
- **ROC Curve Analysis:** A graphical plot illustrating the diagnostic ability of the model across different threshold settings.

## Achievements

- The developed ResNet-50 model achieved a validation accuracy of 97%, indicating its high reliability in detecting IDC regions in breast cancer histopathological images.

## References

- [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
- [Research Papers on IDC Detection](https://www.researchgate.net/publication/263052166_Automatic_detection_of_invasive_ductal_carcinoma_in_whole_slide_images_with_Convolutional_Neural_Networks)
- [SPIE Digital Library](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/9041/1/Automatic-detection-of-invasive-ductal-carcinoma-in-whole-slide-images/10.1117/12.2043872.short#_=_)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2153353922005478)
- [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7787039/)
- [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/9058279)
