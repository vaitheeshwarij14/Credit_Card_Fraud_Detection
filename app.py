import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import requests
import zipfile
from io import BytesIO

# Use secrets from Streamlit Cloud for Kaggle credentials
os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

# Function to authenticate Kaggle API
def authenticate_kaggle():
    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

# Function to download the dataset from Kaggle
def download_dataset():
    dataset_path = "creditcard.csv"
    
    # Check if file exists and remove it if necessary
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
        st.write("Existing dataset found and deleted.")
    
    authenticate_kaggle()
    kaggle_url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=1"

    # Send request to Kaggle URL
    response = requests.get(kaggle_url, stream=True)

    # Check if the response is valid (status code 200)
    if response.status_code == 200:
        try:
            # Try to extract it as a ZIP file
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                zip_ref.extractall()
                st.write("Dataset downloaded and extracted successfully.")
        except zipfile.BadZipFile:
            # If it's not a zip, try to save the CSV directly
            with open(dataset_path, "wb") as f:
                f.write(response.content)
            st.write("Dataset downloaded successfully (not zipped).")
    else:
        st.error("Failed to download the dataset. Please check your Kaggle credentials or internet connection.")

@st.cache_data
def load_data():
    download_dataset()
    
    # Load the dataset
    dataset_path = "creditcard.csv"
    if not os.path.exists(dataset_path):
        st.error(f"{dataset_path} not found after download.")
        return None
    
    # Read the dataset and handle potential parsing issues
    try:
        data = pd.read_csv(dataset_path)
        st.write("Dataset loaded successfully.")
        return data
    except pd.errors.ParserError:
        st.error(f"Failed to parse {dataset_path}. The file may be corrupted.")
        return None

# Load data
data = load_data()

# Ensure data is loaded
if data is not None:
    # Ensure the dataset contains the 'Class' column
    if 'Class' not in data.columns:
        st.error("'Class' column not found in the dataset. Please check the data.")
    else:
        # Separate legitimate and fraudulent transactions
        legit = data[data['Class'] == 0]
        fraud = data[data['Class'] == 1]

        # Undersample legitimate transactions to balance the classes
        legit_sample = legit.sample(n=len(fraud), random_state=2)
        data = pd.concat([legit_sample, fraud], axis=0)

        # Split data into training and testing sets
        X = data.drop(columns="Class", axis=1)
        y = data["Class"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train logistic regression model with increased max_iter
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model performance
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)

        # Create Streamlit app
        st.title("Credit Card Fraud Detection Model")
        st.write(f"Model Training Accuracy: {train_acc:.2f}")
        st.write(f"Model Testing Accuracy: {test_acc:.2f}")
        st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

        # Create input fields for user to enter feature values
        input_df = st.text_input('Input All features as comma-separated values')

        # Create a button to submit input and get prediction
        submit = st.button("Submit")

        if submit:
            try:
                # Get input feature values
                input_df_lst = input_df.split(',')
                features = np.array(input_df_lst, dtype=np.float64)

                # Scale the input features before making predictions
                features = scaler.transform([features])

                # Make prediction
                prediction = model.predict(features)

                # Display result
                if prediction[0] == 0:
                    st.write("Legitimate transaction")
                else:
                    st.write("Fraudulent transaction")
            except ValueError:
                st.write("Please enter valid numerical feature values.")
