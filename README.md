# Bike Sharing Demand Prediction on AWS SageMaker

This project demonstrates an end-to-end Machine Learning workflow to forecast bike rental demand using the XGBoost algorithm on Amazon SageMaker. It utilizes the [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data) dataset from Kaggle.

## Project Overview

The goal is to predict the total count of bikes rented during each hour covered by the test set, based on features like time, weather, and holiday status. The project is structured into three main stages:
1.  **Data Preparation**: Cleaning and feature engineering.
2.  **Model Training**: Training an XGBoost model using SageMaker's built-in algorithm.
3.  **Prediction**: Deploying the model to an endpoint and generating predictions.

## Prerequisites

To run this project, you need:
*   **AWS Account**: With permissions to access SageMaker and S3.
*   **AWS SageMaker Notebook Instance**: Recommended for running the Jupyter notebooks.
*   **S3 Bucket**: To store training data and model artifacts.
*   **Python 3.6+** (Anaconda recommended).
*   **Libraries**: `pandas`, `numpy`, `matplotlib`, `boto3`, `sagemaker`.

## Project Structure

*   `data_preparation.ipynb`: Notebook for data exploration, feature engineering, and preparing datasets for training. It generates `bike_train.csv`, `bike_validation.csv`, and `bike_test.csv`.
*   `xgboost_training.ipynb`: Notebook to upload data to S3, configure the SageMaker XGBoost estimator, train the model, and deploy it to an endpoint.
*   `xgboost_prediction.ipynb`: Notebook to invoke the deployed SageMaker endpoint and generate predictions for the test dataset.
*   `train.csv` & `test.csv`: Raw datasets from Kaggle.
*   `README.md`: This file.

## Usage Guide

Follow these steps to execute the project:

### 1. Data Preparation
Open and run `data_preparation.ipynb`.
*   This notebook loads `train.csv` and `test.csv`.
*   It extracts features like `year`, `month`, `day`, and `hour` from the `datetime` column.
*   It splits the training data into training and validation sets.
*   **Output**: Generates `bike_train.csv`, `bike_validation.csv`, and `bike_test.csv` in the local directory.

### 2. Model Training & Deployment
Open and run `xgboost_training.ipynb`.
*   **Configuration**: Update the `bucket_name` variable with your S3 bucket name.
*   **Upload**: The notebook uploads the generated CSV files to your S3 bucket.
*   **Training**: It initializes a SageMaker XGBoost estimator and trains it on the data in S3.
*   **Deployment**: After training, it deploys the model to a real-time endpoint named `xgboost-biketrain-ver1` (or similar).

### 3. Prediction
Open and run `xgboost_prediction.ipynb`.
*   **Invoke**: Connects to the deployed endpoint.
*   **Predict**: Sends the test data (`bike_test.csv`) to the endpoint in batches.
*   **Result**: Receives predictions, applies `expm1` (inverse of log transformation if applied), and saves the results to `predicted_count_cloud.csv`.

## Dataset Details

The dataset contains the following features:
*   `datetime`: Hourly date + timestamp
*   `season`:  1 = spring, 2 = summer, 3 = fall, 4 = winter
*   `holiday`: whether the day is considered a holiday
*   `workingday`: whether the day is neither a weekend nor holiday
*   `weather`: 
    *   1: Clear, Few clouds, Partly cloudy
    *   2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    *   3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    *   4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
*   `temp`: temperature in Celsius
*   `atemp`: "feels like" temperature in Celsius
*   `humidity`: relative humidity
*   `windspeed`: wind speed
*   `casual`: number of non-registered user rentals initiated
*   `registered`: number of registered user rentals initiated
*   `count`: number of total rentals (Target Variable)

## Clean Up
Remember to delete the SageMaker endpoint and S3 resources after you are done to avoid incurring unexpected charges. You can do this via the AWS Console or using the SageMaker SDK `predictor.delete_endpoint()` method.