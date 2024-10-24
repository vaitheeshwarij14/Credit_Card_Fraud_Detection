import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
import gdown

# Correct Google Drive direct download link
url = "https://drive.google.com/uc?id=1cD78QR_ZJge8XPfrMCYs4rI3s_aOfnLl"
output = "creditcard.csv"
gdown.download(url, output, quiet=False)

# Load the CSV file
try:
    data = pd.read_csv(output)
except Exception as e:
    st.error(f"Error loading the CSV file: {e}")
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

# Create Streamlit app
st.title("Credit Card Fraud Detection Model")
st.write(f"Training accuracy: {train_acc:.2f}, Testing accuracy: {test_acc:.2f}")
st.write("Enter the following features to check if the transaction is legitimate or fraudulent:")

# Create input fields for user to enter feature values
input_df = st.text_input('Input All features (comma-separated values)')

# Split the input by commas and make sure the length matches the number of features
if input_df:
    input_df_lst = input_df.split(',')
    if len(input_df_lst) != X_train.shape[1]:
        st.error(f"Please enter exactly {X_train.shape[1]} feature values.")
    else:
        try:
            # Convert input to NumPy array and reshape for prediction
            features = np.array(input_df_lst, dtype=np.float64)
            submit = st.button("Submit")
            if submit:
                # Make prediction
                prediction = model.predict(features.reshape(1, -1))
                # Display result
                if prediction[0] == 0:
                    st.write("Legitimate transaction")
                else:
                    st.write("Fraudulent transaction")
        except ValueError as ve:
            st.error(f"Invalid input. Ensure all values are numeric. Error: {ve}")
