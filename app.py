import os
import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi

# Streamlit app title
st.title("Credit Card Fraud Detection Model")

def download_dataset():
    """Download the dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    dataset_path = 'mlg-ulb/creditcardfraud'
    
    # Remove existing files if they exist
    if os.path.exists('creditcard.csv'):
        os.remove('creditcard.csv')

    # Download the dataset
    api.dataset_download_files(dataset_path, path='.', unzip=True)

    st.success("Dataset downloaded successfully!")

def load_data():
    """Load the credit card fraud dataset."""
    if not os.path.exists('creditcard.csv'):
        download_dataset()

    try:
        # Load the dataset with handling for bad lines and specifying the correct delimiter
        data = pd.read_csv('creditcard.csv', delimiter=',', on_bad_lines='skip')
        
        # Check if the DataFrame is empty
        if data.empty:
            st.error("The dataset is empty. Please check the downloaded file.")
            return None

        # Display the shape and first few rows of the dataset
        st.write("Dataset loaded successfully!")
        st.write(f"Data shape: {data.shape}")
        st.write(data.head())
        
        # Check for 'Class' column existence
        if 'Class' not in data.columns:
            st.error("The 'Class' column is missing in the dataset. Please check the CSV file.")
            return None
        
        return data
    except Exception as e:
        st.error(f"Failed to parse the CSV file: {e}")
        return None

# Load data and handle potential errors
st.write("Loading the dataset...")
data = load_data()

if data is not None:
    # Proceed with further processing or modeling
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    
    st.write(f"Number of legit transactions: {len(legit)}")
    st.write(f"Number of fraudulent transactions: {len(fraud)}")
