# Deep Learning Assignment 1: Housing Price Prediction

This repository contains a Jupyter notebook that focuses on predicting housing prices using deep learning techniques. The notebook was developed as part of a deep learning assignment and includes various steps such as data preprocessing, model creation, training, and evaluation.

## Description

The **DeepLearning_Assignment1_Housing.ipynb** notebook demonstrates how to build and train a neural network model to predict housing prices. It walks through the entire workflow, starting from data loading, exploration, and preprocessing, all the way through model design, training, and evaluation. The objective is to predict house prices based on various features such as square footage, number of bedrooms, location, and other relevant factors.

### Key Points Covered:
- **Data Loading**: Importing the housing dataset and exploring its structure.
- **Data Preprocessing**: Handling missing data, scaling features, and transforming categorical variables.
- **Model Building**: Creating a deep neural network using Keras/TensorFlow.
- **Model Training**: Training the model with training data, adjusting hyperparameters like learning rate and epochs.
- **Model Evaluation**: Using Mean Squared Error (MSE) and Mean Absolute Error (MAE) to measure the performance of the model.
- **Visualization**: Plotting training history (loss curves) and comparing predicted vs. actual housing prices.

## Notebook Overview

### File
- **DeepLearning_Assignment1_Housing.ipynb**: This notebook implements a neural network model to predict housing prices based on features such as square footage, number of rooms, location, and more.

### Key Sections:
1. **Data Preprocessing**: Loading and cleaning the dataset, handling missing values, and feature scaling.
2. **Model Building**: Creating a neural network model using Keras/TensorFlow.
3. **Model Training**: Training the model using the dataset and fine-tuning hyperparameters.
4. **Evaluation**: Measuring model performance using relevant metrics (e.g., MSE, MAE) and visualizing the results.

## Prerequisites

Before running the notebook, make sure you have the following libraries installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow or Keras
- Pandas
- NumPy
- Matplotlib (for visualization)

You can install all required dependencies by running:

```bash
pip install -r requirements.txt
exit

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/DeepLearning-Housing-Prediction.git

2. Navigate to the directory:
   ```bash
   cd DeepLearning-Housing-Prediction

3. Open the notebook:
   ```bash
   jupyter notebook DeepLearning_Assignment1_Housing.ipynb

Run the cells in sequence to train the model and view the results.
```bash

Results
The notebook demonstrates how to build and train a neural network for housing price prediction. The performance of the model is evaluated based on the test dataset, and various plots are included to visualize the model's predictions versus actual values.

Contributing
Feel free to fork this repository and create pull requests for any improvements, bug fixes, or new features.

License
This project is licensed under the MIT License.
