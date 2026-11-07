# Bank Marketing Classification Project

## 1. Problem Statement

The objective of this project is to build and compare five machine learning classification models to predict whether a bank customer will subscribe to a term deposit based on customer and campaign-related information.

The project also includes an interactive Streamlit application that allows users to upload a test dataset, select a trained classification model, and view its evaluation metrics, confusion matrix, classification report, and predictions.

## 2. Dataset Description

- **Dataset:** Bank Marketing
- **Source:** UCI Machine Learning Repository
- **Dataset URL:** https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Total records:** 45,211
- **Input features used:** 15
- **Target variable:** `y`
- **Target classes:** `yes` and `no`

The target variable indicates whether a customer subscribed to a term deposit.

The `duration` feature was excluded because it represents the duration of the phone call and is not known before contacting the customer. Excluding it helps avoid target leakage when predicting subscription before the call.

## 3. Machine Learning Models Implemented

The following five classification algorithms were implemented using the Bank Marketing dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 4. Data Preprocessing

The following preprocessing techniques were applied:

- Missing numerical values were handled using median imputation.
- Missing categorical values were handled using constant imputation with the value `Unknown`.
- Numerical features were scaled using `StandardScaler`.
- Categorical features were encoded using `OneHotEncoder`.
- The dataset was divided into 80% training data and 20% testing data.
- Stratified sampling was used to preserve the target class distribution.

Preprocessing and classification were combined using Scikit-learn pipelines. The preprocessing steps were fitted on training data and applied to the test data.

The `duration` feature was excluded from the input features.

## 5. Model Evaluation Metrics

All five models were evaluated using the same held-out test dataset containing 9,043 records.

The following six evaluation metrics were calculated:

- Accuracy
- Area Under the ROC Curve (AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

Since the dataset has an imbalanced target distribution, multiple evaluation metrics were considered instead of relying only on accuracy.

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8933 | 0.7115 | 0.6303 | 0.2127 | 0.3180 | 0.3237 |
| KNN | 0.8874 | 0.7027 | 0.5529 | 0.1975 | 0.2911 | 0.2833 |
| Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8932 | 0.7682 | 0.6513 | 0.1871 | 0.2907 | 0.3101 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 89.33% and the highest AUC score of 0.7717. It also achieved the highest precision of 0.6632. However, its recall was relatively low at 0.1786, indicating that it identified only a smaller proportion of actual subscribers.

**2. Decision Tree**

The Decision Tree classifier achieved an accuracy of 89.33%, equal to Logistic Regression. It obtained an F1 score of 0.3180 and an MCC score of 0.3237. Its performance was reasonably balanced compared with Logistic Regression, although its AUC was lower.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 88.74% and an AUC of 0.7027. Its precision was 0.5529 and recall was 0.1975. It showed comparatively lower performance across the evaluation metrics.

**4. Gaussian Naive Bayes**

Naive Bayes achieved an accuracy of 84.52%, which was the lowest among the five models. However, it obtained the highest recall of 0.4641, F1 score of 0.4123, and MCC score of 0.3271. It performed better at identifying the positive class compared with the other classifiers.

**5. Support Vector Machine (SVM)**

SVM achieved an accuracy of 89.32% and an AUC of 0.7682. Its precision was 0.6513, while recall was 0.1871. Its overall performance was comparable to Logistic Regression, but its F1 score was lower than that of Naive Bayes and Decision Tree.

### Overall Winner for the Dataset

Based on the evaluation results, **Gaussian Naive Bayes is the overall preferred model** for this dataset because it achieved the highest recall, F1 score, and MCC.

Although Logistic Regression achieved the highest AUC and Decision Tree achieved the same highest accuracy, Naive Bayes provided better identification of customers who subscribed to the term deposit.

The choice of the best model depends on the evaluation metric and the importance of identifying positive customer responses.

## 6. Project Structure

```text
Bank-Marketing-Classification/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_evaluation_results.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── svm.pkl
```

## 7. GitHub Repository Link

https://github.com/manushareddy28/Bank-Marketing-Classification

The repository contains the Streamlit application, dependency file, dataset, model evaluation results, README, and saved trained model pipelines.

## 8. Streamlit Application

The Streamlit application provides the following features:

- Upload a test dataset in CSV format.
- Select one of the five trained classification models.
- Display all six evaluation metrics.
- Visualize the confusion matrix.
- Display the classification report.
- View prediction results and download predictions.

The application loads the previously saved trained models and their preprocessing pipelines. It evaluates the uploaded test dataset without retraining the models.

**Live Streamlit Application:**

https://bank-marketing-classification-hgxvxa8sub5q2cmcsumvis.streamlit.app/

## 9. Installation and Execution

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 10. Files Included

- `app.py`: Streamlit application source code.
- `requirements.txt`: Required Python packages.
- `README.md`: Project documentation.
- `test_data.csv`: Held-out test dataset containing input features and actual target values.
- `model_evaluation_results.csv`: Comparison of evaluation metrics for all five models.
- `model/`: Directory containing the five saved trained classification pipelines.

## 11. Conclusion

This project demonstrates the implementation, evaluation, and deployment of five machine learning classification algorithms using the Bank Marketing dataset.

The results show that accuracy alone is not sufficient to evaluate classification models on an imbalanced dataset. Recall, F1 score, AUC, and MCC provide additional insight into model performance.

Gaussian Naive Bayes achieved the best recall, F1 score, and MCC among the evaluated models, making it the preferred model for identifying potential term deposit subscribers in this experiment.

**Programme:** BITS Pilani – Work Integrated Learning Programme  
**Assignment:** Machine Learning – Assignment 2
