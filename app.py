import os
import numpy as np
import pandas as pd
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Set Kaggle API credentials from Streamlit secrets
os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

def download_dataset():
    """Download the dataset from Kaggle."""
    api = KaggleApi()
    api.authenticate()

    dataset_path = 'mlg-ulb/creditcardfraud'
    
    # Remove existing dataset if it exists
    if os.path.exists('creditcard.csv'):
        st.write("Existing dataset found and deleted.")
        os.remove('creditcard.csv')

    st.write("Downloading dataset...")
    api.dataset_download_files(dataset_path, path='.', unzip=True)
    st.write("Dataset downloaded successfully.")

def load_data():
    """Load the credit card fraud dataset."""
    if not os.path.exists('creditcard.csv'):
        download_dataset()

    try:
        # Load the dataset
        data = pd.read_csv('creditcard.csv', error_bad_lines=False, warn_bad_lines=True)
        return data
    except Exception as e:
        st.error(f"Failed to parse the CSV file: {e}")
        return None

# Load data
st.title("Credit Card Fraud Detection Model")
st.write("Loading the dataset...")
data = load_data()

if data is not None:
    # Check if the DataFrame is empty
    if data.empty:
        st.error("The dataset is empty. Please check the downloaded file.")
    else:
        # Display dataset info
        st.write(data.info())
        st.write(data.describe())

        # Separate legitimate and fraudulent transactions
        legit = data[data.Class == 0]
        fraud = data[data.Class == 1]

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

        # Train logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Evaluate model performance
        train_acc = accuracy_score(model.predict(X_train), y_train)
        test_acc = accuracy_score(model.predict(X_test), y_test)

        st.write(f"Model Training Accuracy: {train_acc:.2f}")
        st.write(f"Model Testing Accuracy: {test_acc:.2f}")

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
else:
    st.write("Error loading data. Please try again.")
