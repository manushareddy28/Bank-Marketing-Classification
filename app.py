
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("Bank Marketing Classification")
st.write(
    "Upload a test dataset and select a trained machine learning model "
    "to evaluate its performance."
)

# Load saved models
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "KNN": joblib.load("model/knn.pkl"),
        "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
        "SVM": joblib.load("model/svm.pkl")
    }

models = load_models()

# Expected input features
required_features = [
    "age", "job", "marital", "education", "default",
    "balance", "housing", "loan", "contact", "day_of_week",
    "month", "campaign", "pdays", "previous", "poutcome"
]

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

selected_model = st.selectbox(
    "Select a classification model",
    list(models.keys())
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Dataset shape:", data.shape)
    st.dataframe(data.head())

    missing_features = [
        col for col in required_features if col not in data.columns
    ]

    if missing_features:
        st.error(f"Missing required columns: {missing_features}")

    elif "y" not in data.columns:
        st.error(
            "The uploaded CSV must contain the actual target column 'y' "
            "to calculate evaluation metrics."
        )

    else:
        X = data[required_features]
        y_true = data["y"]

        model = models[selected_model]

        y_pred = model.predict(X)

        classifier = model.named_steps["classifier"]

        if hasattr(model, "predict_proba"):
            positive_index = list(classifier.classes_).index("yes")
            y_score = model.predict_proba(X)[:, positive_index]
        else:
            y_score = model.decision_function(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score)
        precision = precision_score(
            y_true, y_pred, pos_label="yes", zero_division=0
        )
        recall = recall_score(
            y_true, y_pred, pos_label="yes", zero_division=0
        )
        f1 = f1_score(
            y_true, y_pred, pos_label="yes", zero_division=0
        )
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader("Model Evaluation Results")
        st.write("Selected model:", selected_model)

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{recall:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred, labels=["no", "yes"])

        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["no", "yes"]
        )
        display.plot(ax=ax, cmap="Blues", values_format="d")
        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.subheader("Classification Report")
        report = classification_report(
            y_true,
            y_pred,
            labels=["no", "yes"],
            zero_division=0,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose())

        # Predictions
        st.subheader("Predictions")
        output = data.copy()
        output["predicted_y"] = y_pred
        st.dataframe(output.head(20))

        csv = output.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download predictions",
            data=csv,
            file_name="bank_marketing_predictions.csv",
            mime="text/csv"
        )
