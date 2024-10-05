import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("static/accepted_2007_to_2018Q4.csv")

# Streamlit app title
st.title("Good Loans and Bad Loans")

# Dataset Overview
st.title("Dataset Overview For Good/Bad Loan")
st.write(df)

# Features Overview
st.title("Dataset's Features Overview For Good/Bad Loan")
st.write("'term' : The number of payments on the loan, where values are in months and can be either 36 or 60.")
st.write("'int_rate :  The interest rate on the loan")
st.write("'sub_grade  : Assigned loan subgrade score based on borrower's credit history")
st.write("'emp_length': Borrower's employment length in years.")
st.write("'dti' : A ratio calculated using the borrower's total monthly debt payments on the total debt obligations, excluding mortgage, divided by the borrower's monthly income")
st.write("'mths_since_recent_inq': Months since most recent inquiry")
st.write("'revol_util' : Revolving line utilization rate, or the amount of credit the borrower uses relative to all available revolving credit")
st.write("'bc_util': Ratio of total current balance to high credit/credit limit for all bankcard accounts")
st.write("'num_op_rev_tl' : Number of open revolving accounts")

# Data observation
st.title("Observation")
st.write("We have a lot of features but we got this top 9 features using Logistic Regression with SequentialFeatureSelector")

# Filter the DataFrame for relevant loan statuses
df = df[(df['loan_status'] == 'Fully Paid') | (df['loan_status'] == 'Charged Off')]

# Selected final features for modeling
final_features = ['term', 'int_rate', 'sub_grade', 'emp_length', 'dti', 'mths_since_recent_inq', 'revol_util', 'bc_util', 'num_op_rev_tl', 'loan_status']
df = df[final_features]

# Preprocessing
df_temp = df.copy()
df_temp['term'] = df_temp['term'].apply(lambda x: int(x[0:3]))  # Extract numeric part
df_temp['loan_status'] = df_temp['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

mapp = {'< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
df_temp['emp_length'] = df_temp['emp_length'].map(mapp)

# Fill missing values
df_temp.fillna(0, inplace=True)

# Drop rows with emp_length <= 0
df_temp = df_temp[df_temp['emp_length'] > 0]

# One-hot encoding for categorical variables
df_temp = pd.get_dummies(df_temp)

# Show Preprocessed Data + Feature Engineering
st.title("Preprocessed Data + Feature Engineering")
st.write(df_temp)

# Splitting the data
X = df_temp.drop(["loan_status"], axis=1)
y = df_temp["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building and evaluating ML models
st.title("We have built different ML models to see their performance")

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest Classifier": RandomForestClassifier(),
    "KNeighbors Classifier": KNeighborsClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=200)
}

# Train Models and Store Accuracies
accuracy_results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy_results[model_name] = accuracy_score(pred, y_test) * 100

# Display Model Accuracies
st.title("Model Performance")
for model_name, accuracy in accuracy_results.items():
    st.write(f"{model_name}: Accuracy of {accuracy:.2f}%")
    st.write("*" * 20)

# Input Interface for Predictions
st.title("Loan Prediction Input")
term = st.selectbox("Loan Term (in months)", (36, 60))
int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
sub_grade = st.selectbox("Subgrade", df['sub_grade'].unique())
emp_length = st.selectbox("Employment Length", list(mapp.keys()))
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=10.0)
mths_since_recent_inq = st.number_input("Months Since Recent Inquiry", min_value=0, value=0)
revol_util = st.number_input("Revolving Utilization", min_value=0.0, value=10.0)
bc_util = st.number_input("Bankcard Utilization", min_value=0.0, value=10.0)
num_op_rev_tl = st.number_input("Number of Open Revolving Accounts", min_value=0, value=1)

# Prediction Button
if st.button("Predict Loan Status"):
    # Create DataFrame for input
    input_data = pd.DataFrame({
        'term': [term],
        'int_rate': [int_rate],
        'sub_grade': [sub_grade],
        'emp_length': [mapp[emp_length]],  # Map employment length to numerical value
        'dti': [dti],
        'mths_since_recent_inq': [mths_since_recent_inq],
        'revol_util': [revol_util],
        'bc_util': [bc_util],
        'num_op_rev_tl': [num_op_rev_tl]
    })

    # One-hot encode input data
    input_data = pd.get_dummies(input_data)
    # Ensure input_data has the same columns as training data
    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[X.columns]

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make Predictions
    model_to_use = models["Random Forest Classifier"]  # You can change this to any model you want to use for prediction
    prediction = model_to_use.predict(input_data_scaled)

    # Display Result
    loan_status = "Good Loan" if prediction[0] == 1 else "Bad loan"
    st.write(f"The predicted loan status is: **{loan_status}**")
