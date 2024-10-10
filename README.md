# Deep Learning Assignment 1: Housing Price Prediction

This repository contains a Jupyter notebook that focuses on predicting housing prices using deep learning techniques. The notebook was developed as part of a deep learning assignment and includes various steps such as data preprocessing, model creation, training, and evaluation.

## Description

The **DeepLearning_Assignment1_Housing.ipynb** notebook demonstrates how to build and train a neural network model to predict housing prices. It walks through the entire workflow starting from data loading, exploration, and preprocessing, all the way through model design and evaluation. The objective is to predict house prices based on a variety of features such as square footage, number of bedrooms, location, and other relevant factors. 

Key points covered:
- Handling missing data
- Feature scaling and normalization
- Building a deep learning model using Keras/TensorFlow
- Model performance evaluation using loss functions like Mean Squared Error (MSE)
- Visualization of training history and prediction results

## Dataset Overview

### Features

- **Price**: The target variable representing the price of the house (in currency).
- **Area**: The total area of the house in square feet.
- **Bedrooms**: The number of bedrooms in the house.
- **Bathrooms**: The number of bathrooms in the house.
- **Stories**: The number of stories (levels) in the house.
- **Main Road**: A binary variable indicating whether the property is located on a main road (1 for yes, 0 for no).
- **Guest Room**: A binary variable indicating the availability of a guest room (1 for yes, 0 for no).
- **Basement**: A binary variable indicating whether the house has a basement (1 for yes, 0 for no).
- **Hot Water Heating**: A binary variable indicating whether the house has hot water heating (1 for yes, 0 for no).
- **Air Conditioning**: A binary variable indicating whether the house has air conditioning (1 for yes, 0 for no).
- **Parking**: A binary variable indicating the availability of parking spaces (1 for yes, 0 for no).
- **Preferred Area**: A binary variable indicating whether the house is located in a preferred area (1 for yes, 0 for no).
- **Furnishing Status**: A categorical variable indicating the furnishing status of the house (e.g., furnished, semi-furnished, unfurnished).

### Dataset Size

The dataset contains a total of **545 records** (houses) and **12 features** (including the target variable). Each record corresponds to a specific house with its respective attributes.
### File
- **DeepLearning_Assignment1_Housing.ipynb**: This notebook implements a neural network model to predict housing prices based on features such as square footage, number of rooms, location, and more.

### Key Sections:
1. **Data Preprocessing**: Loading and cleaning the dataset, handling missing values, and feature scaling.
2. **Model Building**: Creating a neural network model using Keras/TensorFlow (or relevant library).
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

### You can install all required dependencies by running:

	pip install -r requirements.txt

### 1. Clone the repository:
	git clone https://github.com/your-username/DeepLearning-Housing-Prediction.git

### 2. Navigate to the directory:
   	cd DeepLearning-Housing-Prediction

### 3. Open the notebook:
   	jupyter notebook DeepLearning_Assignment1_Housing.ipynb

Run the cells in sequence to train the model and view the results.

### Results
	The notebook demonstrates how to build and train a neural network for housing price prediction. The performance of the model is evaluated based on the test dataset, and various plots are included to visualize the model's predictions versus actual values.

### Contributing
	Feel free to fork this repository and create pull requests for any improvements, bug fixes, or new features.

### License
	This project is licensed under the MIT License.
