import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Set the title for the app
st.title("Classifying Fraudulent and Valid Transactions")

# Load the dataset
data = pd.read_csv('static/fraud.csv')

# Display the dataset overview
st.header("Dataset Overview")
st.write(data)

# Display dataset column descriptions
st.subheader("Dataset Column Descriptions")
st.write("""
- **step**: Integer. Maps a unit of time in the real world. In this case, 1 step is 1 hour of time, with a total of 744 steps (30 days simulation).
- **type**: String/categorical. Type of transaction: CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
- **amount**: Float. Amount of the transaction in local currency.
- **nameOrig**: String. Customer who initiated the transaction.
- **oldbalanceOrg**: Float. Initial balance before the transaction.
- **newbalanceOrig**: Float. New balance after the transaction.
- **nameDest**: String. Customer who is the recipient of the transaction.
- **oldbalanceDest**: Float. Initial balance of the recipient before the transaction.
- **newbalanceDest**: Float. New balance of the recipient after the transaction.
- **isFraud**: Boolean/binary. Indicates if the transaction is fraudulent (1) or valid (0).
- **isFlaggedFraud**: Boolean/binary. Indicates if the transaction is flagged as fraudulent (1) or not flagged (0). A transaction is flagged if it's fraudulent and involves a transfer over 200,000 in local currency.
""")

# Data preprocessing
# Drop unnecessary column if it exists
if 'Unnamed: 0' in data.columns:
    data.drop('Unnamed: 0', axis=1, inplace=True)

# Map 'type' column to numerical values
type_mapping = {"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5}
data["type"] = data["type"].map(type_mapping)

# Drop rows with NaN values in 'type' column after mapping
data.dropna(subset=['type'], inplace=True)

# Convert 'type' column to integer type
data['type'] = data['type'].astype(int)

# Encode 'isFraud' column to numerical format (0 and 1)
data["isFraud"] = data["isFraud"].map({0: 0, 1: 1})

# Prepare features and target variables for the model
X = np.array(data[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']])
y = np.array(data['isFraud'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Display the model's accuracy
st.header("Decision Tree Accuracy on Test Data:")
st.write(model.score(X_test, y_test))

# Function to calculate diffOrg and suspectRatio
def calculate_anomaly_metrics(row):
    diffOrg = row['oldbalanceOrg'] - row['newbalanceOrig'] + row['amount']
    suspectRatio = (row['oldbalanceOrg'] + row['amount']) / row['newbalanceDest'] if row['newbalanceDest'] != 0 else np.nan
    return diffOrg, suspectRatio

# Prediction section
st.header("Transaction Prediction")
st.subheader("Enter Transaction Details")

# User inputs for prediction
type_input = st.selectbox("Transaction Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
amount_input = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg_input = st.number_input("Old Balance (Origin)", min_value=0.0)
newbalanceOrig_input = st.number_input("New Balance (Origin)", min_value=0.0)
newbalanceDest_input = st.number_input("New Balance (Destination)", min_value=0.0)

# Button to make prediction
if st.button("Predict Transaction"):
    # Ensure the inputs are logical
    if amount_input > oldbalanceOrg_input:
        st.warning("The transaction amount cannot exceed the old balance.")
    else:
        # Map the type input to numerical value
        type_numeric = type_mapping[type_input]
        
        # Create features array for prediction
        features = np.array([[type_numeric, amount_input, oldbalanceOrg_input, newbalanceOrig_input]])
        
        # Make prediction
        prediction = model.predict(features)
        prediction_result = "Fraud" if prediction[0] == 1 else "Valid"
        
        # Calculate diffOrg and suspectRatio for the input
        diffOrg, suspectRatio = calculate_anomaly_metrics({
            'oldbalanceOrg': oldbalanceOrg_input,
            'newbalanceOrig': newbalanceOrig_input,
            'amount': amount_input,
            'newbalanceDest': newbalanceDest_input
        })
        
        # Display prediction result and anomaly metrics
        st.write(f"Prediction Result: The transaction is classified as: **{prediction_result}**")
        st.write(f"Calculated diffOrg: {diffOrg}")
        st.write(f"Calculated suspectRatio: {suspectRatio}")
        
        # Display anomaly check results
        if diffOrg < 0:
            st.write("Anomaly Check Result: Potential Fraud (Negative diffOrg)")
        if suspectRatio > 1:
            st.write("Anomaly Check Result: Potential Fraud (suspectRatio > 1)")

