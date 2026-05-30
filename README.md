# ML Image Classifier (CNN)

## Overview
This project implements a Machine Learning image classifier using a Convolutional Neural Network (CNN) trained on the classic CIFAR-10 dataset. The model utilizes a progressive feature-extraction architecture with sequential convolutional layers using 32, 64, and 128 filters.

### Dataset Profile
* **Name**: CIFAR-10
* **Volume**: 60,000 low-resolution color images (50,000 train / 10,000 test)
* **Target Classes**: 10 distinct categories (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks)
* **Dimensions**: 32x32 pixels with 3 color channels (RGB)

### Model Architecture
The network processes spatial data hierarchically, moving from simple edges to complex object shapes:

1. **Convolutional Block 1 (32 Filters)**: Extracts low-level features like edges, lines, and basic textures.
2. **Convolutional Block 2 (64 Filters)**: Combines low-level features to detect mid-level shapes, corners, and textures.
3. **Convolutional Block 3 (128 Filters)**: Captures complex, high-level structural features specific to the target classes.

Each block utilizes $3\times3$ kernels, ReLU activation functions, Max Pooling ($2\times2$) to reduce spatial dimensions, and Dropout to prevent overfitting.

[Input: 32x32x3]│[Conv2D: 32 filters]  ──> [MaxPooling] ──> [Dropout]│[Conv2D: 64 filters]  ──> [MaxPooling] ──> [Dropout]│[Conv2D: 128 filters] ──> [MaxPooling] ──> [Dropout]│[Flatten] ──> [Dense Layer] ──> [Dropout] ──> [Softmax Output (10 Classes)]

### Training Strategy
* **Loss Function**: Sparse Categorical Cross-Entropy
* **Optimizer**: Adam (Adaptive Moment Estimation)
* **Regularization**: Dropout layers (20%–50% rate) to mitigate overfitting on the small input dimensions.

## Technologies
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## Features
- Image preprocessing
- CNN model training
- Model evaluation

## How It Works
1. Load and preprocess the dataset
2. Train the CNN model
3. Evaluate model performance

### Prerequisites
Make sure you have Python 3.8+ installed on your system.

### Installation
1. Clone the repository:
   git clone https://github.com/tony-lam-nun/ml-image-classifier
   
3. Install the required dependencies:
   pip install -r requirements.txt

## Usage

### Training the Model:
   python train.py, and python model.py

### Evaluating the Model
    python evaluate.py

## Results and performance
After training for 20 epochs, the model achieves the following performance metrics on the test dataset:
### Training Accuracy | ~76.2% |
### Validation Accuracy | ~74.0% |
### Training Loss | ~65.0% |
### Validation Loss | ~72.0% |

### Training History
<p align="center">
  <img src="assets/accuracy_plot.png" width="45%" alt="Model Accuracy">
  <img src="assets/loss_plot.png" width="45%" alt="Model Loss">
</p>

## Future Improvements
### Improve model performance
### Increase dataset size
### Add web interface




