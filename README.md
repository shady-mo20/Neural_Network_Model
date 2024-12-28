# Neural_Network_Model

This repository contains an implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras for classifying handwritten digits from the MNIST dataset.

## Project Overview

The project demonstrates how to:

- Preprocess and normalize the MNIST dataset.
- Build and train a CNN model.
- Evaluate the model's accuracy on the test set.

## Model Architecture

The CNN model has the following architecture:

1. **Convolutional Layer:** 32 filters, kernel size (3, 3), ReLU activation.
2. **MaxPooling Layer:** Pool size (2, 2).
3. **Convolutional Layer:** 64 filters, kernel size (3, 3), ReLU activation.
4. **MaxPooling Layer:** Pool size (2, 2).
5. **Flatten Layer**: Converts 2D data to 1D.
6. **Dense Layer:** 64 neurons, ReLU activation.
7. **Output Layer:** 10 neurons (for 10 classes), softmax activation.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shady-mo20/Neural_Network_Model.git
   ```
2. Install the required dependencies:
   ```bash
   pip install tensorflow
   ```

## Usage

Run the script to train and evaluate the model:

```bash
python model.py
```

## Dataset

The MNIST dataset is automatically downloaded using TensorFlow's built-in functions. It contains 60,000 training images and 10,000 testing images of handwritten digits (0-9).

## Results

- **Training Accuracy:** ~99% after 5 epochs.
- **Test Accuracy:** Achieves high accuracy on the unseen test dataset.

## License

This project is licensed under the MIT License. Feel free to use and modify the code.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.
