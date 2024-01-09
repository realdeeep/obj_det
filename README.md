# Sign Language Recognition using Convolutional Neural Networks

## Overview

This project aims to detect and recognize American Sign Language gestures in real-time using a Convolutional Neural Network (CNN). The project consists of two main components: 

1. **Training the CNN Model:** A Jupyter Notebook (`main1.ipynb`) is provided to train a CNN model on the Sign Language MNIST dataset. The notebook covers data preprocessing, model building, training, and saving the trained model.

2. **Live Gesture Detection:** An OpenCV-based Python script (`live_detect.py`) utilizes the trained CNN model to perform live sign language gesture detection through your computer's webcam.

## CNN Model Training

### Dependencies
- Python 3.x
- TensorFlow
- Keras
- NumPy
- pandas
- Jupyter Notebook

### Usage
1. Open and run the `main1.ipynb` Jupyter Notebook to train the CNN model.
2. Save the trained model for future use.

## Live Gesture Detection

### Dependencies
- Python 3.x
- OpenCV
- NumPy
- Keras

### Usage
1. Ensure the trained model is saved in the same directory as the Python script.
2. Run the `live_detect.py` script.
3. Use your computer's webcam to perform real-time sign language gesture detection.

## Important Notes
- The model is trained on the Sign Language MNIST dataset, and labels correspond to letters A to Z.
- Adjust the frame size, preprocessing steps, and label mapping as needed for optimal performance.

## Credits
- The Sign Language MNIST dataset is used for training. Dataset source: [Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist)
