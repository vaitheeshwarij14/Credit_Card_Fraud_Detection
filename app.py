import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import streamlit as st
import requests
from io import StringIO

# Google Drive file URL (replace FILE_ID with your actual file ID)
file_url = "https://drive.google.com/uc?export=download&id=FILE_ID"

@st.cache_data
def load_data():
    response = requests.get(file_url)
    file_content = StringIO(response.text)
    return pd.read_csv(file_content)

# Load data
data = load_data()

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
