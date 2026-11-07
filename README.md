
# Bank Marketing Classification Project

## 1. Project Overview

This project implements and compares five machine learning classification
algorithms to predict whether a bank customer will subscribe to a term deposit.

The project also includes a Streamlit application that allows users to upload
a CSV dataset, select a trained model, and view its evaluation results.

## 2. Dataset

- Dataset: Bank Marketing
- Source: UCI Machine Learning Repository
- Dataset URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Total records: 45,211
- Input features used: 15
- Target variable: y
- Target classes: yes and no

The duration feature was excluded because it represents call duration,
which is not known before contacting the customer.

## 3. Machine Learning Algorithms

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

The preprocessing steps include:

- Handling missing numerical values using median imputation.
- Handling missing categorical values using constant imputation.
- Scaling numerical features using StandardScaler.
- Encoding categorical features using OneHotEncoder.
- Splitting the dataset into 80% training data and 20% testing data.
- Using stratified sampling to preserve class proportions.

Preprocessing and classification are combined into pipelines.
The pipelines are saved along with the trained models.

## 5. Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

The dataset has an imbalanced target distribution, so multiple
evaluation metrics are considered instead of relying only on accuracy.

## 6. Project Structure

Bank_Marketing_ML_Project/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model_evaluation_results.csv
|
|-- model/
    |-- logistic_regression.pkl
    |-- decision_tree.pkl
    |-- knn.pkl
    |-- naive_bayes.pkl
    |-- svm.pkl

## 7. Streamlit Application

The application provides the following features:

- CSV file upload.
- Selection of one of the five trained models.
- Calculation of six evaluation metrics.
- Confusion matrix visualization.
- Classification report.
- Prediction results with CSV download.

The application loads saved models and does not retrain them.

## 8. Installation and Execution

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## 9. Files Included

- app.py: Streamlit application.
- requirements.txt: Required Python packages.
- test_data.csv: Test dataset with actual target values.
- model_evaluation_results.csv: Model comparison results.
- model/: Saved trained machine learning pipelines.

## 10. Author

Machine Learning Assignment 2
BITS Pilani - Work Integrated Learning Programme
