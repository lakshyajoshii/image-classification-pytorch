# Image Classification with PyTorch

This project implements an image classification model using **PyTorch**. The goal of the project is to build a neural network that classifies images from the CIFAR-10 dataset with high accuracy. The model architecture utilizes advanced techniques such as **data augmentation** and **transfer learning** to improve performance, achieving an accuracy of 92%.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The project is designed to classify images from the **CIFAR-10** dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes include objects like airplanes, cars, birds, cats, and others. The main focus is to build an efficient and accurate model using **PyTorch**, while incorporating real-world use cases like fraud detection and customer segmentation in the fintech domain.

## Features
- **Custom Neural Network**: Built from scratch using PyTorch.
- **Transfer Learning**: Fine-tuned a pre-trained model to boost accuracy.
- **Data Augmentation**: Enhanced training data to avoid overfitting.
- **92% Accuracy**: Achieved through careful tuning of hyperparameters.
- **Applications**: Can be adapted for real-world problems like fraud detection and customer segmentation.

## Installation
To run the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/lakshyajoshii/image-classification-pytorch.git
    cd image-classification-pytorch
    ```

2. Set up a virtual environment and install the required dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3. Download the CIFAR-10 dataset (this is handled automatically in the code if the dataset is not present).

## Usage
1. To train the model, run:
    ```bash
    python train.py
    ```

2. To evaluate the model on the test dataset, run:
    ```bash
    python evaluate.py
    ```

3. You can also load the pre-trained model and use it for inference on custom images:
    ```bash
    python inference.py --image <path_to_image>
    ```

## Model Architecture
The custom neural network model consists of the following layers:
- **Convolutional layers**: For feature extraction.
- **Batch Normalization and Dropout**: For regularization and improving model generalization.
- **Fully connected layers**: For classification.

The model also incorporates **transfer learning** from a pre-trained ResNet, improving performance on small datasets.

## Results
- **Training Accuracy**: 92%
- **Test Accuracy**: 91%
- **Loss**: The model converged to a low loss after several epochs of training.

Here’s a sample of the classification performance on the test set:

| Class        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| Airplane     | 0.93      | 0.91   | 0.92     |
| Automobile   | 0.95      | 0.94   | 0.94     |
| Bird         | 0.89      | 0.87   | 0.88     |
| ...          | ...       | ...    | ...      |

## Contributing
Contributions are welcome! If you want to contribute to this project, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
