# 🚲 Bike Sharing Demand Prediction

A machine learning regression project that predicts the number of bikes rented based on time, weather, and environmental conditions.

The project compares Linear Regression and Random Forest Regressor models and uses a preprocessing pipeline with One-Hot Encoding and Standard Scaling.

---

## 📌 Project Overview

Bike-sharing systems generate demand patterns based on factors such as:

- Time of day
- Month
- Year
- Weather conditions
- Temperature
- Humidity
- Wind speed
- Working days
- Holidays

The goal of this project is to build a machine learning model that can predict the total number of bike rentals (`cnt`).

This project focuses on understanding the complete machine learning workflow, from data preprocessing to model evaluation and interpretation.

---

## 🎯 Objective

Build a regression model capable of predicting bike-sharing demand and compare different machine learning approaches.

The main objectives were:

- Analyze the dataset
- Identify numerical and categorical features
- Prepare the data for machine learning
- Preserve the chronological structure of the dataset
- Build preprocessing pipelines
- Train multiple regression models
- Evaluate model performance
- Perform hyperparameter tuning
- Compare tuned and original models
- Analyze feature importance
- Save the final trained model

---

## 📊 Dataset

The dataset contains **17,379 observations** of bike-sharing demand.

### Target Variable

`cnt`

Represents the total number of bike rentals.

### Features Used

#### Categorical Features

- `season`
- `yr`
- `mnth`
- `hr`
- `holiday`
- `weekday`
- `workingday`
- `weathersit`

#### Numerical Features

- `temp`
- `atemp`
- `hum`
- `windspeed`

### Removed Columns

The following columns were removed:

- `instant` — record index
- `dteday` — date column
- `casual` — casual users
- `registered` — registered users

`casual` and `registered` were removed because they directly contribute to the target `cnt` and could cause data leakage.

---

## 🔧 Data Preprocessing

The dataset contains both categorical and numerical features.

A `ColumnTransformer` was used to apply different preprocessing operations.

### Categorical Features

Categorical features were transformed using:

```python
OneHotEncoder(handle_unknown="ignore")
