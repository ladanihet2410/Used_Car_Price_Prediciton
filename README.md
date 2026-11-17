# 🚗 Used Car Price Prediction System

An end-to-end **Machine Learning regression project** that predicts the estimated resale price of a used car from its specifications.

The project demonstrates a complete ML workflow: loading data, preprocessing numerical and categorical features, comparing regression algorithms, cross-validation, hyperparameter tuning, evaluating the final model, saving the trained pipeline, and using it in a Streamlit web application.

> **Dataset note:** The included `dummy_used_cars.csv` contains 300 rows of synthetic/demo data. It is generated data, not real car-market listings. Results from this dataset are only for learning and testing the code. Do not present its performance as real-world accuracy or business impact.

---

## 📌 Project Overview

When buying or selling a used car, estimating a reasonable resale price can be difficult. A car's value depends on several factors, such as its brand, manufacturing year, kilometres driven, fuel type, transmission, ownership history, engine capacity, and seating capacity.

This project uses supervised machine learning to learn relationships between these input features and the car's selling price.

### 🎯 Objectives

- Build a regression model to estimate used-car resale prices.
- Prepare numerical and categorical data for machine learning.
- Compare multiple regression algorithms using cross-validation.
- Tune a Random Forest model using `GridSearchCV`.
- Evaluate the selected model on a separate test dataset.
- Save the complete preprocessing-and-model pipeline.
- Provide a simple interface for making predictions with Streamlit.

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Loading and manipulating tabular data |
| NumPy | Numerical calculations |
| Scikit-learn | Preprocessing, model training, tuning, and evaluation |
| Joblib | Saving and loading the trained ML pipeline |
| Streamlit | Building the interactive web application |

---

## 📂 Project Structure

```text
Used-Car-Price-Prediction/
│
├── dummy_used_cars.csv       # Demo dataset
├── train_model.py            # Data preprocessing, training and evaluation
├── app.py                    # Streamlit prediction application
├── requirements.txt          # Python dependencies
├── used_car_price_model.pkl  # Generated after model training
└── README.md                 # Project documentation
```

`used_car_price_model.pkl` is generated when `train_model.py` runs. You can regenerate it whenever you retrain the model.

---

## 📊 Dataset Description

The training script reads `dummy_used_cars.csv`.

- **Rows:** 300
- **Columns:** 9
- **Target variable:** `selling_price`
- **Problem type:** Supervised Machine Learning — Regression

### Features

| Feature | Description | Example |
|---|---|---|
| `brand` | Manufacturer of the car | Maruti, Hyundai, Honda |
| `year` | Manufacturing year | 2019 |
| `km_driven` | Total kilometres driven | 45,000 |
| `fuel` | Fuel category | Petrol, Diesel, CNG |
| `transmission` | Gear transmission type | Manual, Automatic |
| `owner` | Previous ownership category | First, Second, Third |
| `engine_cc` | Engine capacity in cubic centimetres | 1200 |
| `seats` | Seating capacity | 5 |
| `selling_price` | Target resale price in Indian rupees | 450000 |

### Target Variable

The model predicts `selling_price`.

The other eight columns are used as input features (`X`), while `selling_price` is the output (`y`).

### Important Dataset Limitation

The supplied CSV is synthetic demo data. It is useful for understanding the ML workflow and checking that the project runs, but it does not represent actual market prices.

For a stronger portfolio project:

1. Find a real used-car dataset with a licence that permits your intended use.
2. Record its source, author/publisher, licence, currency, region, and access date.
3. Map its columns to the feature names expected by `train_model.py`.
4. Check missing values, duplicate records, invalid prices, and unusual values.
5. Do not upload a dataset to GitHub if its licence does not allow redistribution.

---

## ⚙️ Machine Learning Workflow

### 1. Load the dataset

Pandas loads the CSV file. The script separates the independent variables (`X`) from the target (`y`).

### 2. Data preprocessing

The script uses Scikit-learn pipelines and a `ColumnTransformer` to apply different preprocessing steps to numerical and categorical columns.

**Numerical features**
- `year`
- `km_driven`
- `engine_cc`
- `seats`

Steps:
- `SimpleImputer(strategy="median")` fills missing numerical values using the median.
- `StandardScaler()` standardizes numerical features.

**Categorical features**
- `brand`
- `fuel`
- `transmission`
- `owner`

Steps:
- `SimpleImputer(strategy="most_frequent")` fills missing categorical values using the most frequent category.
- `OneHotEncoder(handle_unknown="ignore")` converts categories into numerical columns and safely handles categories not seen during fitting.

**Why use a pipeline?**

A pipeline keeps preprocessing and model training together. During cross-validation, preprocessing is fitted separately within each training fold, helping prevent information from the validation fold from leaking into preprocessing.

### 3. Train-test split

The dataset is split using:

- 80% training data
- 20% testing data
- `random_state=42` for reproducibility

The test set is held aside for final evaluation.

### 4. Compare regression algorithms

The training script compares these four models:

| Algorithm | Basic idea |
|---|---|
| Linear Regression | Models a linear relationship between input features and price |
| Ridge Regression | Linear regression with L2 regularization to help control model complexity |
| Random Forest Regressor | Combines predictions from multiple decision trees |
| Gradient Boosting Regressor | Builds trees sequentially, with later trees learning from earlier errors |

The models are compared using **5-fold cross-validation** on the training portion of the data. The comparison metric is Mean Absolute Error (MAE).

### 5. Hyperparameter tuning

The script uses `GridSearchCV` to tune the Random Forest pipeline.

It searches over:

| Parameter | Values tested |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 10, 20 |
| `min_samples_leaf` | 1, 2 |

The tuned Random Forest's cross-validation MAE is compared with the cross-validation MAE of the candidate models. The script selects the candidate with the lower MAE.

### 6. Model evaluation

The selected model is evaluated on the held-out test set using three regression metrics.

**Mean Absolute Error (MAE)**

The average absolute difference between actual and predicted prices. It is expressed in the same currency unit as the target.

**Root Mean Squared Error (RMSE)**

The square root of the average squared prediction error. It penalizes larger errors more strongly than MAE.

**R-squared (R²)**

Measures how much of the target variation is explained by the model relative to predicting the test-set mean. It can be negative when the model performs worse than that baseline.

> The training script prints the actual test metrics when you run it. No performance figures are stated here because they should be measured from the exact dataset and environment you use.

### 7. Save the trained model

The selected complete pipeline is saved as:

```text
used_car_price_model.pkl
```

It contains both preprocessing and the trained estimator. The Streamlit app loads this file and uses it to make predictions.

---

## 🖥️ Streamlit Application

The `app.py` file provides an interactive form where a user can enter:

- Car brand
- Manufacturing year
- Kilometres driven
- Fuel type
- Transmission
- Previous owner category
- Engine capacity
- Number of seats

After clicking **Predict car price**, the app sends these values to the saved model pipeline and displays the estimated resale price in Indian rupees.

The application is an educational demonstration. A prediction is not a guaranteed market valuation; actual resale prices also depend on condition, variant, location, service history, accident history, and market demand. Some of these factors are not included in the current dataset.

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME
```

Or download the repository as a ZIP and extract it.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train and evaluate the model

```bash
python train_model.py
```

This command:
1. Loads `dummy_used_cars.csv`.
2. Preprocesses numerical and categorical features.
3. Compares four regression algorithms using 5-fold cross-validation.
4. Tunes Random Forest with `GridSearchCV`.
5. Selects a model based on cross-validation MAE.
6. Evaluates it on the held-out test set using MAE, RMSE, and R².
7. Saves `used_car_price_model.pkl`.

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser if it does not open automatically.

---

## 🧠 Machine Learning Concepts Demonstrated

- Supervised learning
- Regression problem formulation
- Numerical and categorical feature handling
- Missing-value imputation
- Feature scaling
- One-hot encoding
- Scikit-learn `Pipeline` and `ColumnTransformer`
- Train-test split
- K-fold cross-validation
- Model comparison
- Hyperparameter tuning with `GridSearchCV`
- Regression evaluation: MAE, RMSE, and R²
- Model persistence with Joblib
- Model inference through Streamlit

---

## 🔍 Current Scope and Future Improvements

The current version is a learning project using a synthetic dataset and a defined set of car features. Possible next improvements include:

- Train and validate using a real, appropriately licensed dataset.
- Add exploratory data analysis (EDA) and visualizations.
- Add more useful vehicle details, such as model/variant, location, and vehicle condition, if reliable data is available.
- Investigate outliers and residual errors.
- Evaluate performance across different price ranges and car categories.
- Compare additional algorithms and tuning strategies.
- Add prediction explanations and feature-importance analysis.
- Improve input validation and user experience.
- Deploy the app to a hosting platform.

These are suggested future improvements; they are not all implemented in the current code.

---

## ⚠️ Reproducibility and Model File Note

Scikit-learn model files saved with Joblib/pickle can be version-sensitive. The uploaded `used_car_price_model.pkl` was created with Scikit-learn 1.5.1, while the current inspection environment has Scikit-learn 1.8.0 and could not load that artifact because of an internal compatibility change.

**Recommended:** install the dependencies in a clean environment, run `python train_model.py` to regenerate the model file, and then run the app in that same environment. For reproducible sharing, pin the versions you successfully tested in `requirements.txt`.

Do not load pickle/joblib files from untrusted sources.

---

## 📬 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Het%20Ladani-blue?logo=linkedin)](https://linkedin.com/in/het-ladani-5001bb29a/)
[![GitHub](https://img.shields.io/badge/GitHub-ladanihet2410-black?logo=github)](https://github.com/ladanihet2410)



## 📄 Disclaimer

This project is developed for educational and portfolio purposes. The included synthetic dataset and resulting predictions must not be treated as real market valuations or financial advice.
