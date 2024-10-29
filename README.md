# Image-Classification-Using-CNN 

## Project Overview

This project uses a Convolutional Neural Network (CNN) model to classify images from the CIFAR-10 dataset, a popular dataset in image recognition research. The dataset includes 60,000 color images in 10 classes, with 6,000 images per class. The goal is to train a model to accurately classify unseen images into one of these classes.

## Dataset

The dataset is split into 50,000 training images and 10,000 test images, with labels indicating the class for each image.

## Project Structure

1. Data Loading and Preprocessing
- Loaded the CIFAR-10 dataset directly from TensorFlow's keras.datasets.
- Normalized pixel values for improved model convergence and performance.
  
2. Model Building
- Constructed a CNN model using TensorFlow and Keras.
- Designed the model with several convolutional layers, max pooling, and dense layers to improve feature extraction and classification.
- Used activation functions such as ReLU in hidden layers and Softmax in the output layer for multi-class classification.

  
3. Model Training and Evaluation
- Compiled the model with categorical cross-entropy loss and used the Adam optimizer.
- Trained the model on the training dataset with validation on the test set.
- Evaluated model accuracy and loss, visualizing performance metrics to assess classification accuracy.

## Results

The model demonstrates a strong ability to classify CIFAR-10 images into the correct categories. Performance may be further enhanced by fine-tuning hyperparameters, adding data augmentation, or expanding the model architecture.

## Conclusion

This CNN model serves as an effective image classification tool for small image datasets like CIFAR-10. With further improvements, it can be adapted to handle more complex and higher-resolution image data.
