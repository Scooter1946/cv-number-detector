# Number Detector

## Overview
This application implements a real-time handwritten digit recognition system using OpenCV and PyTorch. It captures a live video feed, detects and isolates a paper region, and feeds the processed image into a Convolutional Neural Network (CNN) trained on the MNIST dataset to classify digits (0-9) with high accuracy.


## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Scooter1946/numberDetector.git
```

### 2. Set Up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
Before running the detector, you need to train the neural network to generate the `model.pth` file.
```bash
python train_model.py
```

### 5. Run the Application
Start the live camera feed and detection system.
```bash
python main.py
```

## Data Flow
The following diagram illustrates how data moves through the application, from the camera lens to the final prediction:

`Webcam Feed` -> `Image Preprocessing (Grayscale, Blur, Canny)` -> `Paper Detection (Contour Finding)` -> `Perspective Transform (Warp & Crop)` -> `Adaptive Thresholding (Clean Up)` -> `Neural Network (CNN)` -> `Prediction (0-9)`

## Key Features of The CNN Architecture
*   **Convolutional Layers (`nn.Conv2d`)**: These are the "eyes" of the network. They slide small filters (kernels) across the input image to create feature maps. Early layers might detect simple lines or edges, while deeper layers combine these to recognize complex shapes like loops or intersections.
*   **ReLU Activation (`F.relu`)**: This function introduces non-linearity into the model. Without it, the network would just be a linear regression model. ReLU allows the network to learn complex, non-linear relationships between pixels.
*   **Max Pooling (`F.max_pool2d`)**: This down-samples the feature maps, reducing their size while keeping the most important information (the strongest features). This makes the model more computationally efficient and helps it focus on the *presence* of a feature rather than its exact pixel location (making it translation invariant).
*   **Dropout (`nn.Dropout`)**: To prevent the model from simply memorizing the training data (overfitting), dropout randomly "turns off" a percentage of neurons during training. This forces the network to learn redundant pathways and robust features, ensuring it performs well on new, messy handwriting it hasn't seen before.
*   **Fully Connected Layers (`nn.Linear`)**: These are the final decision makers. They take the high-level abstract features extracted by the convolutional layers and map them to the 10 possible output classes (digits 0-9).
