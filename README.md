# Rossmann Sales Prediction Project

This repository contains an analysis and prediction project for the Rossmann Sales dataset. The project leverages time-series analysis and LSTM (Long Short-Term Memory) models to predict future sales for the Rossmann drug store chain.

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset Description](#dataset-description)
- [Methods Used](#methods-used)
- [Training](#Training)
- [Usage](#usage)

## Project Overview

The primary goal of this project is to predict daily sales for over 1,000 Rossmann stores based on historical sales data. The project includes exploratory data analysis (EDA), data preprocessing, and building a predictive model using LSTM networks. 

Key objectives include:
- Understanding sales trends over time and across different stores.
- Evaluating the impact of promotions and holidays on sales.
- Developing a robust LSTM model for accurate sales forecasting.
- Visualizing the actual and predicted sales to assess model performance.

## Repository Structure

The repository contains the following files:

- `rossman.ipynb`: The main Jupyter Notebook containing all the steps of the analysis, including data exploration, preprocessing, model building, and evaluation.
- `rossmanOutput.pdf`: A PDF export of the Jupyter Notebook with all the outputs and visualizations.
- `train.csv`: The training dataset, containing historical sales data for Rossmann stores.
- `test.csv`: The test dataset, used to predict future sales (without the sales column).
- `store.csv`: Additional information about each store, including store type, assortment, competition distance, and promotion details.

## Dataset Description

The dataset includes three files:

### Train Dataset (`train.csv`):
- **Store**: Unique identifier for each store.
- **Date**: The date of the sales record.
- **Sales**: Sales made on that day (target variable).
- **Customers**: Number of customers visiting the store on that day.
- **Open**: Whether the store was open (1) or closed (0).
- **Promo**: Whether a promotion was active on that day.
- **StateHoliday**: State holiday indicator ('a', 'b', 'c', '0' for no holiday).
- **SchoolHoliday**: Whether the store was affected by a school holiday.

### Test Dataset (`test.csv`):
- Contains similar features as the train dataset but does not include the `Sales` column, which is the target for prediction.

### Store Dataset (`store.csv`):
- **Store**: Unique identifier for each store (links with train/test datasets).
- **StoreType**: Categorizes stores into four types: a, b, c, d.
- **Assortment**: Level of product assortment: a, b, c.
- **CompetitionDistance**: Distance to the nearest competitor store.
- **CompetitionOpenSince[Month/Year]**: Approximate opening date of the nearest competitor.
- **Promo2**: Continuous promotion indicator (0 = no, 1 = yes).
- **Promo2Since[Year/Week]**: Start date of Promo2 for each store.
- **PromoInterval**: Months when Promo2 is active.

## Methods Used

### Exploratory Data Analysis (EDA)

EDA involves summarizing the main characteristics of the data and visualizing trends, patterns, and relationships. In this project, EDA is performed to:
- Examine the sales trends over time.
- Identify the effects of holidays and promotions on sales.
- Explore how different stores perform relative to each other.

Key tools and techniques used for EDA include:
- **Pandas** for data manipulation and summarization.
- **Matplotlib** and **Seaborn** for creating visualizations to uncover trends and patterns.

### Data Preprocessing

Before building the model, data needs to be preprocessed to ensure it is clean and in a suitable format for analysis. Steps include:
- Handling missing values.
- Merging different datasets (train, test, and store).
- Filtering out closed stores and sorting the data by date.
- Normalizing the data using Min-Max Scaling to bring all features into the same scale, which is crucial for training neural networks.

### Long Short-Term Memory (LSTM) Networks

LSTM is a type of recurrent neural network (RNN) that is well-suited for time-series forecasting tasks. Unlike traditional RNNs, LSTMs can learn long-term dependencies and handle the vanishing gradient problem, making them effective for capturing patterns in sequential data over extended periods.

**Key aspects of the LSTM model in this project**:
- **Sequential Data Handling**: LSTM models are designed to process and predict sequences of data, making them ideal for time-series forecasting where the order of data points is significant.
- **Model Architecture**: The model consists of multiple layers:
  - **Two LSTM Layers**: 
    - The first LSTM layer with 100 units returns sequences (`return_sequences=True`), which allows stacking another LSTM layer on top.
    - The second LSTM layer also with 100 units but doesn't return sequences (`return_sequences=False`), producing an output that goes to the next layer.
  - **Dropout Layer**: Added with a dropout rate of 0.2 to prevent overfitting by randomly setting a fraction of input units to 0 during each update cycle.
  - **Dense Layers**: 
    - A Dense layer with 50 units and ReLU activation function helps in adding non-linearity and learning complex relationships.
    - The final Dense layer with 1 unit is used to output the predicted sales value.

The LSTM model in this project is used to:
- Predict future sales based on past sales data.
- Capture the complex temporal dependencies in the sales data, such as trends and seasonality.

### Training

The model is trained over 200 epochs, which means the model sees the entire training dataset 200 times. This extensive training helps the model to deeply learn the patterns in the data over long periods.

**Early Stopping** is employed to prevent overfitting. It stops the training process if the model's performance on a validation set stops improving for 40 consecutive epochs (patience set to 40). This technique not only reduces overfitting but also saves training time by stopping early if further training does not yield better results.


### Evaluation Metrics

The model's performance is evaluated using the Root Mean Squared Error (RMSE), which measures the differences between predicted and actual sales values. Lower RMSE values indicate better model performance. 

## Usage

To explore and run the analysis, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/micdwill/Sales-Forecasting.git
   cd Sales-Forecasting
