# DigiYouth II - Big Data and Artificial Intelligence Project - Team 2

This team project is part of the DigiYouth II – ReGeneration Academy on Big Data & Artificial Intelligence (powered by Microsoft). 

## Description

The project involves building a model that accurately predicts the prices of an AirBnB listing in Athens. The dataset includes various information 
about the listings, such as the amount of people it accommodates or the number of bathrooms it has. The goal is to process the dataset and keep the 
most useful features, train a model to accurately predict the prices of the listings and finally deploy the model.

## Dataset

The dataset is provided in two file categories called listings.csv and reviews.csv.

The first contains nominal information about the listings, like its neighbourhood, its description, amenities, bedrooms, bathrooms etc. and is the one used to develop our model fro price prediction.

The other file-group contains various ratings for the listings above in free text.

## Code Structure

### Notebooks

The initial Exploratory Data Analysis and Modeling was done through notebooks which are stored in the notebooks folder. Through these notebooks we
were able to identify the useful features of the dataset, prepare the data for training by preprocessing them and finally tune the hyperparameters of the models.

### Model

The final trained model, with tuned hyperparameters, is stored in the models folder.

### Python Scripts

The preprocessing step is included in the src folder and contains the functions that process the original dataset, performing the necessary transformations to it, making it ready to be used by the model.

Additionally, the server that uses the model to make the predictions for new establishments through FastAPI is contained in the src folder. A user can send a 
post request to the server with the information of his listing and get a price estimation as result.

### Docker Files

A Docker file for building an image that contains the server is also included, as well as a text file with the necessary Python package requirements.

## Dependencies

Development was done on Python 3.8 and the packages used were:

- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib
- xgboost
- joblib
- fastapi
- uvicorn
- pydantic

## Usage

### Use deployed model

If you want to deploy the model and use it to make predictions throught the API, here are the steps:

1. Build the Docker image using the dockerfile and the requirements text file.
2. Create a container with the image.
3. Either use the container locally or upload it to Azure so it can be used more easily.

### Use the model on the dataset

1. Load the dataset from 'listings.csv'.
2. Process it by using the preprocess_dataset function of preprocess.py.
3. Load the model.
4. Use the model on the dataset as you see fit.
