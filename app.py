import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import gdown

# Function to load data from Google Drive
def load_data_from_drive(drive_url):
    # Extract the file ID from the Google Drive URL
    file_id = drive_url.split('/')[-2]
    # Construct the gdown URL
    gdown_url = f"https://drive.google.com/file/d/1cD78QR_ZJge8XPfrMCYs4rI3s_aOfnLl/view?usp=drive_link"
    # Download the CSV file
    data = pd.read_csv(gdown_url)
    return data

# Load data from Google Drive link
drive_link = st.text_input("Enter your Google Drive CSV file link:")
if drive_link:
    try:
        data = load_data_from_drive(drive_link)
        st.write("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    st.warning("Please enter a valid Google Drive link to load the dataset.")
    st.stop()

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

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(model.predict(X_train), y_train)
test_acc = accuracy_score(model.predict(X_test), y_test)

# Display model performance metrics
st.write("Training Accuracy:", train_acc)
st.write("Testing Accuracy:", test_acc)

# Create input fields for user to enter feature values
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")
input_df = st.text_input('Input All features separated by commas:')
input_df_lst = input_df.split(',')

# Create a button to submit input and get prediction
submit = st.button("Submit")

if submit:
    # Ensure the input has the correct number of features
    if len(input_df_lst) != X.shape[1]:
        st.error(f"Please enter exactly {X.shape[1]} feature values.")
    else:
        # Get input feature values
        try:
            features = np.array(input_df_lst, dtype=np.float64)
            # Make prediction
            prediction = model.predict(features.reshape(1, -1))
            # Display result
            if prediction[0] == 0:
                st.write("Legitimate transaction")
            else:
                st.write("Fraudulent transaction")
        except ValueError as ve:
            st.error(f"Error in input values: {ve}")
