import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import pickle
import streamlit as st
import plotly.graph_objects as go
import os

# Custom LabelEncoder that can handle unseen labels
class CustomLabelEncoder:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.classes = None

    def fit(self, data):
        self.encoder.fit(data)
        self.classes = set(self.encoder.classes_)

    def transform(self, data):
        # Replace unseen labels with a placeholder
        new_data = [x if x in self.classes else "unseen" for x in data]
        return self.encoder.transform(new_data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

# Part 1: Data Preprocessing and Model Training
def preprocess_data(train_path, test_path):
    st.text("Starting data preprocessing...")
    
    # Load datasets
    df_train = pd.read_csv(train_path)
    df_train["source"] = "train"
    df_test = pd.read_csv(test_path)
    df_test["source"] = "test"
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    st.text(f"Loaded {len(df_train)} training samples and {len(df_test)} test samples.")

    # Fill missing values and convert to numeric
    df['Credit_Product'] = df['Credit_Product'].fillna("NA")
    df["Is_Active"] = df["Is_Active"].replace({"Yes": 1, "No": 0}).astype(float)
    
    st.text("Filled missing values and converted 'Is_Active' to numeric.")

    # Custom Label encoding for categorical columns
    cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product']
    le_dict = {}
    for col in cat_cols:
        le = CustomLabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    st.text(f"Applied custom label encoding to {', '.join(cat_cols)}.")

    # Separate train and test
    df_train = df[df["source"] == "train"].drop(columns=['ID', 'source'])
    df_test = df[df["source"] == "test"].drop(columns=['ID', 'source'])
    
    st.text("Preprocessing completed.")
    return df_train, df_test, le_dict

def train_model(df):
    st.text("Starting model training...")
    
    # Separate features and target
    X = df.drop(columns=['Is_Lead'])
    y = df['Is_Lead']
    
    st.text(f"Prepared features (shape: {X.shape}) and target (shape: {y.shape}).")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.text(f"Split data into train (80%) and test (20%) sets.")

    # Standardize features
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    st.text("Standardized features using StandardScaler.")

    # Train the model
    lgb_params = {
        'learning_rate': 0.045,
        'n_estimators': 1000,  # Reduced for quicker training
        'max_bin': 84,
        'num_leaves': 10,
        'max_depth': 20,
        'reg_alpha': 8.457,
        'reg_lambda': 6.853,
        'subsample': 0.749
    }
    model = LGBMClassifier(**lgb_params)
    st.text("Training LightGBM model...")
    model.fit(X_train_scaled, y_train)
    st.text("Model training completed.")

    # Evaluate the model
    y_pred = model.predict_proba(X_test_scaled)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred)
    st.text(f"Model evaluation - ROC AUC Score: {roc_auc:.4f}")

    return model, sc

def save_model_and_preprocessors(model, le_dict, sc):
    st.text("Saving model and preprocessors...")
    with open('models/cross_sell_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('models/label_encoders.pkl', 'wb') as file:
        pickle.dump(le_dict, file)
    with open('models/standard_scaler.pkl', 'wb') as file:
        pickle.dump(sc, file)
    st.text("Model and preprocessors saved successfully.")

# Part 2: Streamlit App
@st.cache_resource
def load_model():
    st.text("Loading saved model and preprocessors...")
    with open('models/cross_sell_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('models/label_encoders.pkl', 'rb') as file:
        le_dict = pickle.load(file)
    with open('models/standard_scaler.pkl', 'rb') as file:
        sc = pickle.load(file)
    st.text("Model and preprocessors loaded successfully.")
    return model, le_dict, sc

def preprocess_inputs(data, le_dict, sc):
    df = pd.DataFrame([data])
    cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product']
    for col in cat_cols:
        df[col] = le_dict[col].transform(df[col])
    df = pd.DataFrame(sc.transform(df), columns=df.columns)
    return df

def main():
    st.title('Cross-sell Prediction App')

    # Check if model exists, if not, train and save it
    if not (os.path.exists('models/cross_sell_model.pkl') and
            os.path.exists('models/label_encoders.pkl') and
            os.path.exists('models/standard_scaler.pkl')):
        st.warning("Model or preprocessors not found. Training a new model...")
        try:
            df_train, _, le_dict = preprocess_data('static/train_s3TEQDk.csv', 'static/test_mSzZ8RL.csv')
            model, sc = train_model(df_train)
            save_model_and_preprocessors(model, le_dict, sc)
            st.success("Model trained and saved successfully!")
        except Exception as e:
            st.error(f"An error occurred during model training: {str(e)}")
            return
    else:
        st.info("Loading existing model...")
        try:
            model, le_dict, sc = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"An error occurred while loading the model: {str(e)}")
            return

    # User input form
    st.header('Enter Customer Information')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', min_value=18, max_value=100, value=30)
    region_code = st.text_input('Region Code')
    occupation = st.text_input('Occupation')
    channel_code = st.text_input('Channel Code')
    vintage = st.number_input('Vintage', min_value=0, value=10)
    credit_product = st.selectbox('Credit Product', ['Yes', 'No', 'NA'])
    avg_account_balance = st.number_input('Average Account Balance', min_value=0.0, value=1000.0)
    is_active = st.selectbox('Is Active', ['Yes', 'No'])

    if st.button('Predict'):
        user_inputs = {
            'Gender': gender,
            'Age': age,
            'Region_Code': region_code,
            'Occupation': occupation,
            'Channel_Code': channel_code,
            'Vintage': vintage,
            'Credit_Product': credit_product,
            'Avg_Account_Balance': avg_account_balance,
            'Is_Active': 1 if is_active == 'Yes' else 0
        }

        # Debugging output
        #st.write("User inputs:", user_inputs)

        processed_inputs = preprocess_inputs(user_inputs, le_dict, sc)
        prediction = model.predict_proba(processed_inputs)[0][1]

        st.subheader('Prediction Result')
        st.write(f'The probability of cross-sell is: {prediction:.2f}')

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Cross-sell Probability"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.75], 'color': "gray"},
                    {'range': [0.75, 1], 'color': "darkgray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.75}}))
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
