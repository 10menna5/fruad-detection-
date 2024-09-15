#!pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
import numpy as np
from scipy import stats




df = pd.read_csv('fraudTrain.csv')
df1 = pd.read_excel("fraudTest.xlsx")



def preprocessing(df):
    df = df.drop(columns=['lastName'])
    df = df.drop(columns=['firstName'])
    df = df.drop(columns=['merchant'])

    # df.duplicated().sum() there is not duplicated rows

    # Fill missing values
    # Fill missing values with mean for 'Amount' and median for 'is_fraud'

    df['Amount'] = df['Amount'].fillna(df['Amount'].median())
    df['trans_num'] = df['trans_num'].fillna(df['trans_num'].mode()[0])
    df['is_fraud'] = df['is_fraud'].fillna(df['is_fraud'].median())
    print("Missing values filled with median")

    #df['Amount'].fillna(df['Amount'].mean(), inplace=True)
    #df['trans_num'].fillna(df['trans_num'].mode()[0], inplace=True)
    #df['is_fraud'].fillna(df['is_fraud'].mean(), inplace=True)
    #print("Missing values filled with mean and median")

    # Create new features
    # mean of amount for one customer

    # Calculate difference between current transaction amount and average transaction amount

    df['Diff_Amount_Mean'] = df['Amount'] - (df.groupby('Card Number')['Amount'].transform('mean'))

    # Sort dataframe by 'Card Number' and 'Time'
    df = df.sort_values(by=['Card Number', 'Time'])

    # Calculate time difference between current and previous transactions for each card
    df['Time'] = pd.to_datetime(df['Time'])  # convert time from object to float
    df['Time_Diff_Prev_Trans'] = df.groupby('Card Number')['Time'].diff().dt.seconds / 60
    df['Time_Diff_Prev_Trans'] = df['Time_Diff_Prev_Trans'].fillna(0)
    df.sort_values(by='ID', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.sort_index()
    pd.to_datetime(df['Time'])

    df['year'] = df['Time'].dt.year
    df['month'] = df['Time'].dt.month
    df['day'] = df['Time'].dt.day
    df['hour'] = df['Time'].dt.hour
    df['minute'] = df['Time'].dt.minute
    df = df.drop(columns=['Time'])
    print("create feature done")
    '''
    # ________
    # outliers
    # قائمة بأسماء الأعمدة
    columns = ['Card Number']

    # حلقة لتطبيق الخوارزمية على كل عمود
    # التعليق: حساب الربع الأول والثالث
    for col in columns:
        # حساب الربع الأول والثالث
        q1 = np.percentile(df[col], 25)
        q3 = np.percentile(df[col], 75)

        # طباعة بيانات df[col]
        print(df[col])

        # حساب نطاق القيم الشاذة
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # التعرف على القيم الشاذة
        lower_outliers = df[df[col] < lower_bound]
        upper_outliers = df[df[col] > upper_bound]

        # طباعة عدد القيم الشاذة
        outliers_count = len(lower_outliers) + len(upper_outliers)
        print(f"The number of outliers in {col}: {outliers_count}")

        # استبدال القيم الشاذة بالقيم المعدلة
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    print("Outliers and categorical preprocessing done")

    print("outliers done")
    '''
    # 2feature scalling
    
    #columns_to_scale = ["Amount", "Card Number", "Diff_Amount_Mean", "Diff_Amount_Mean"]
    #scaler = StandardScaler()

    #df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    #print("scalling done")
    from sklearn.preprocessing import MinMaxScaler

    columns_to_scale = ["Card Number"]
    scaler = MinMaxScaler()

    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    print("Scalingdone")
    # ________
    # 3categorical data
    cate = df.select_dtypes(include=['object'])
    # category
    for col in cate.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("categorical done")
    return df

df = preprocessing(df)
df1 = preprocessing(df1)

#print(df.info())




df.to_csv('data_train.csv', index=False)
df1.to_csv('data_test.csv', index=False)
'''
df = pd.read_csv('data_train.csv')
df1 = pd.read_csv('data_test.csv')
'''
#print(df["is_fraud"].value_counts())
#print(df1["is_fraud"].value_counts())

X_train = df.drop(columns=['is_fraud'])
#1X_train = df[['Diff_Amount_Mean','Amount','category']]
y_train = df['is_fraud']

# 1X_train= df[['Diff_Amount_Mean','Amount','category']]
X_test = df1.drop(columns=['is_fraud'])
y_test = df1['is_fraud']


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
X_test, y_test = smote.fit_resample(X_test, y_test)





#print(" after")
#print(y_train.value_counts())
#print(y_test.value_counts())



#y_train = y_train.astype(int)

correlation_matrix = X_train.corrwith(y_train)
#print("Correlation with is_fraud:")
#print(correlation_matrix)

print("=============================================.")
'''
# Define the hyperparameter grid
param_grid = {'C': [0.1, 1, 10, 100]}  # Regularization parameter
svm = SVC()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=15)

# Perform grid search cross-validation
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred)
print("SVM Accuracy:",accuracy1)
'''


# Support Vector Machine

def SVMf():
    svm_model = SVC( probability=True)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)

    # Evaluate Support Vector Machine
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_precision = precision_score(y_test, svm_predictions)
    svm_recall = recall_score(y_test, svm_predictions)
    svm_f1 = f1_score(y_test, svm_predictions)

    print("\nSupport Vector Machine Evaluation:")
    print("Accuracy:", round(svm_accuracy, 7))
    print("Precision:", round(svm_precision, 7))
    print("Recall:", round(svm_recall, 7))
    print("F1 Score:", round(svm_f1, 7))
#SVMf()
print("=============================================.")
# LogisticRegression_model
def LogisticRegressionf():


    log_model = LogisticRegression(solver='liblinear', C=0.0001, random_state=0)
    log_model.fit(X_train, y_train)
    y_pred = log_model.predict(X_test)
    y_pred_train = log_model.predict(X_train)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, y_pred_train)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("LogisticRegression Evaluation:")
    print("Accuracy: ", accuracy)
    #print("Accuracy_train: ", accuracy_train)
    print("Recall Score: ", recall)
    print("Precision Score: ", precision)
    print("F1Score:",f1)

LogisticRegressionf()

# Random Forest
print("=============================================.")
def RandomForestf():

    # Create the model with 90 trees and adjusted parameters
    RF_model = RandomForestClassifier(n_estimators=7, criterion='entropy',
                                      random_state=7)

    # Fit on training data
    RF_model.fit(X_train, y_train)
    rf_predictions = RF_model.predict(X_test)
    y_pred_train = RF_model.predict(X_train)
    # Evaluate Random Forest
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    true_positives = sum((y_test == 1) & (rf_predictions == 1))
   # print("true_positives",true_positives)
    false_positives = sum((y_test == 0) & (rf_predictions == 1))
    #print("false_positives",false_positives)
    # Calculate True Negatives and Total Samples
    total_samples = len(y_test)
    #print("total_samples", total_samples)
    true_negatives = total_samples - (true_positives + false_positives)
    #print("true_negatives", true_negatives)

    # Calculate Precision
    precision_manual = true_positives / (true_positives + false_positives)

    # Calculate Accuracy manually
    accuracy_manual = (true_positives + true_negatives) / total_samples



    accuracy_train = accuracy_score(y_train, y_pred_train)
    # Calculate True Positives and False Positives

    # Calculate Precision


    rf_recall = recall_score(y_test, rf_predictions)
    rf_f1 = f1_score(y_test, rf_predictions)

    print("\nRandom Forest Evaluation:")
    print("Accuracy:", rf_accuracy)
    #print("Accuracy_train: ", accuracy_train)
    print("Precision:", precision_manual)
    print("Recall:", rf_recall)
    print("F1Score:",rf_f1)
RandomForestf()

# ---------------------------------------------------------------------------------
# BoosXGt
print("=============================================.")
def BoosXGtf():
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

    print("\nXGBoost Evaluation:")
    print("Accuracy:", xgb_accuracy)
    #print("Accuracy_train: ", accuracy_train)
    print("Precision:", xgb_precision)
    print("Recall:", xgb_recall)
    print("F1 Score:", xgb_f1)
BoosXGtf()
print("=============================================.")
#KMeans
def KMeansf():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    # Initialize and fit the KMeans model
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(X_train)

    # Calculate Inertia
    inertia = kmeans_model.inertia_
    print("Inertia (SSE):", inertia)

    # Calculate Silhouette Score
    #silhouette = silhouette_score(X_train, kmeans_model.labels_)
    #print("Silhouette Score:", silhouette)
KMeansf()

