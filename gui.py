import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score

from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier

def train_logistic_regression(X_train, y_train, X_test, y_test):
    log_model = LogisticRegression(solver='liblinear', C=0.0001, random_state=0)
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    y_pred_train = log_model.predict(X_train)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "Accuracy": accuracy,
        #"Accuracy_train": accuracy_train,
        "Recall": recall,
        "Precision": precision,
        "F1Score": f1
    }
def random_forest(X_train, y_train, X_test, y_test):
    # Create Random Forest model with specified parameters
    RF_model = RandomForestClassifier(n_estimators=7, criterion='entropy',
                                      random_state=7)

    # Fit on training data
    RF_model.fit(X_train, y_train)
    rf_predictions = RF_model.predict(X_test)
    y_pred_train = RF_model.predict(X_train)
    # Evaluate Random Forest
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    rf_precision = precision_score(y_test, rf_predictions)
    rf_recall = recall_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)

    return {
        "Accuracy": rf_accuracy,
        #"Accuracy_train: ": accuracy_train,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1 Score": rf_f1
    }
def SVM(X_train, y_train, X_test, y_test):
    # Create SVM model
    svm_model = SVC( probability=True)

    # Fit model on training data
    svm_model.fit(X_train, y_train)

    # Make predictions on test data
    svm_predictions = svm_model.predict(X_test)

    # Evaluate SVM model
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions)
    svm_recall = recall_score(y_test, svm_predictions)
    svm_f1 = f1_score(y_test, svm_predictions)

    return {
        "Accuracy": svm_accuracy,
        "Precision": svm_precision,
        "Recall": svm_recall,
        "F1 Score": svm_f1
    }
def XGBoost(X_train, y_train, X_test, y_test):
    xgb_model = XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    y_pred_train = xgb_model.predict(X_train)
    # Evaluate XGBoost
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    accuracy_train = accuracy_score(y_train, y_pred_train)
    xgb_precision = precision_score(y_test, xgb_predictions)
    xgb_recall = recall_score(y_test, xgb_predictions)
    xgb_f1 = f1_score(y_test, xgb_predictions)
    return {
        "Accuracy": xgb_accuracy,
        #"Accuracy_train: ": accuracy_train,
        "Precision": xgb_precision,
        "Recall": xgb_recall,
        "F1 Score": xgb_f1
    }
def K_Means(X_train):

    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X_train)

    inertia = kmeans_model.inertia_

    #silhouette = silhouette_score(X_train, kmeans_model.labels_)

    return {
        "Inertia (SSE)": inertia,
        #"Silhouette Score": silhouette
    }

# GUI code
st.title("Fraud Detection Project Using AI Classification")
st.sidebar.title("Choose Model")

# Load data
df_train = pd.read_csv('data_train.csv')
df_test = pd.read_csv('data_test.csv')

# Preprocess data

X_train = df_train.drop(columns=['is_fraud'])
#X_train = df_train[['Diff_Amount_Mean','Amount','month' ]]
y_train = df_train['is_fraud']

X_test = df_test.drop(columns=['is_fraud'])
#X_test = df_test[['Diff_Amount_Mean' ,'Amount' ,'month']]
y_test = df_test['is_fraud']

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
X_test, y_test = smote.fit_resample(X_test, y_test)


model_option = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Support Vector Machine" ,
     "xgboost"  , "kmeans"])

if model_option == "Logistic Regression":
    st.write(" Logistic Regression:", train_logistic_regression(X_train, y_train, X_test, y_test))

elif model_option == "Random Forest":
    st.write(" Random Forest:", random_forest(X_train, y_train, X_test, y_test))

elif model_option == "xgboost":
    st.write(" xgboost:",  XGBoost(X_train, y_train, X_test, y_test))

elif model_option == "kmeans":
    st.write(" kmeans:", K_Means(X_train))

elif model_option == "Support Vector Machine":
    st.write(" Support Vector Machine:", SVM(X_train, y_train, X_test, y_test))
