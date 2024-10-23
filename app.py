import os
import pandas as pd
import streamlit as st
from io import BytesIO
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Download dataset from Kaggle and validate it
def download_dataset():
    # Specify the dataset path
    dataset_path = "creditcard.csv"

    # Authenticate and download using Kaggle API
    authenticate_kaggle()
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset ZIP file
    st.write("Downloading dataset from Kaggle...")
    api.dataset_download_files('mlg-ulb/creditcardfraud', path='./', unzip=True)

    # Validate the downloaded file
    if not os.path.exists(dataset_path):
        st.error(f"Failed to find {dataset_path} after downloading.")
        return None

    # Check the file size to ensure it's not corrupted
    file_size = os.path.getsize(dataset_path)
    if file_size == 0:
        st.error("The downloaded CSV file is empty. Please check the dataset.")
        return None

    st.write("Dataset downloaded and validated successfully.")
    return dataset_path

# Load the data from the unzipped CSV
def load_data():
    dataset_path = "creditcard.csv"  # Change this if your dataset has a different name

    # Check if dataset exists, if not, download it
    if not os.path.exists(dataset_path):
        dataset_path = download_dataset()
        if dataset_path is None:
            return None

    # Attempt to read the dataset
    try:
        st.write("Loading the dataset...")
        with open(dataset_path, 'r') as f:
            # Preview the first few lines of the dataset
            preview = f.readlines()[:5]
            st.write("Preview of the dataset:")
            for line in preview:
                st.write(line.strip())
        
        data = pd.read_csv(dataset_path, encoding='utf-8')

        # Check for the 'Class' column
        if 'Class' not in data.columns:
            st.error("The 'Class' column is missing in the dataset. Please check the CSV file.")
            return None

        st.write("Dataset loaded successfully.")
        return data
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty. Please check the dataset.")
        return None
    except pd.errors.ParserError:
        st.error("Failed to parse the CSV file. The file may be corrupted.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Main Streamlit app logic
def main():
    st.title("Credit Card Fraud Detection Model")
    
    # Load data
    data = load_data()

    if data is None:
        st.error("Error loading data. Please try again.")
        return
    
    st.write("Data Summary:")
    st.write(data.describe())
    
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
