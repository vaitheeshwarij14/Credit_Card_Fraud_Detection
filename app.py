import os
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Authenticate Kaggle credentials using Streamlit secrets
def authenticate_kaggle():
    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

# Download dataset from Kaggle and unzip it
@st.cache_data
def download_dataset():
    dataset_path = "creditcard.csv"
    
    # Remove existing dataset if found
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
        st.write("Existing dataset found and deleted.")
    
    # Authenticate and download using Kaggle API
    authenticate_kaggle()
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset ZIP file
    st.write("Downloading dataset from Kaggle...")
    api.dataset_download_file('mlg-ulb/creditcardfraud', file_name='creditcard.csv', path='./')
    
    # Extract the ZIP file
    zip_file_path = "creditcardfraud.zip"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # Delete the zip file after extraction
    os.remove(zip_file_path)
    
    st.write("Dataset downloaded and extracted successfully.")
    
# Load the data from the unzipped CSV
@st.cache_data
def load_data():
    dataset_path = "creditcard.csv"
    
    # Check if dataset exists, if not, download it
    if not os.path.exists(dataset_path):
        download_dataset()

    try:
        # Load the CSV into a pandas DataFrame
        return pd.read_csv(dataset_path)
    except pd.errors.ParserError:
        st.error("Failed to parse the CSV file. The file may be corrupted.")
        return None

# Main Streamlit app logic
def main():
    st.title("Credit Card Fraud Detection Model")
    
    # Load data
    data = load_data()

    if data is None:
        st.error("Error loading data. Please try again.")
        return
    
    # Separate legitimate and fraudulent transactions
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    
    # Undersample legitimate transactions to balance the classes
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # Split data into features and target
    X = balanced_data.drop(columns="Class", axis=1)
    y = balanced_data["Class"]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate model performance
    train_acc = accuracy_score(model.predict(X_train), y_train)
    test_acc = accuracy_score(model.predict(X_test), y_test)

    st.write(f"Model Training Accuracy: {train_acc:.2f}")
    st.write(f"Model Testing Accuracy: {test_acc:.2f}")
    
    # Input section for user to test the model
    st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
    input_df = st.text_input('Input all features as comma-separated values')

    submit = st.button("Submit")

    if submit:
        try:
            input_df_lst = input_df.split(',')
            features = np.array(input_df_lst, dtype=np.float64).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            prediction = model.predict(features_scaled)

            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        except ValueError:
            st.write("Please enter valid numerical feature values.")

# Run the main app
if __name__ == "__main__":
    main()
