import os
import joblib
import pandas as pd
import streamlit as st
from typing import Any, Dict, List
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants for directories and file names
MODEL_DIR = 'models'
DATA_DIR = 'datasets'
DATA_FILE = 'cleaned_transaction_dataset.csv'
MODEL_NAMES = [
    'LGBM Classifier', 
    'XGBoost Classifier', 
    'AdaBoost Classifier',
]

# Load dataset
data_path = os.path.join(DATA_DIR, DATA_FILE)
df = pd.read_csv(data_path)

# Load models
def load_models(model_names: List[str]) -> Dict[str, Any]:
    """Load machine learning models from disk."""
    models = {}
    for name in model_names:
        path = os.path.join(MODEL_DIR, f"{name.replace(' ', '')}.joblib")
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model {name}: {str(e)}")
    return models

models = load_models(MODEL_NAMES)

# Prepare features and target
X = df.drop(columns=['FLAG'])
y = df['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Prediction and metrics evaluation function
def calculate_metrics(y_true, y_pred, average_type='binary'):
    """Calculate and return accuracy, recall, F1, and precision scores."""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average=average_type)
    f1 = f1_score(y_true, y_pred, average=average_type)
    prec = precision_score(y_true, y_pred, average=average_type)
    return acc, rec, f1, prec

def load_and_predict(sample):
    try:
        # Using StandardScaler to scale numric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        sample_trans = scaler.transform(sample)

        # Using SMOTE to handle class imbalance
        X_resampled, y_resampled = SMOTE(random_state=123).fit_resample(X_train_scaled, y_train)

        results = []
        for name, model in models.items():
            y_resampled_pred = model.predict(X_resampled)
            flag_pred = model.predict(sample_trans)
            acc, rec, f1, prec = calculate_metrics(y_resampled, y_resampled_pred)

            results.append({
                'Model': name,
                'Predicted Fraud': 'Yes' if flag_pred[0] == 1 else 'No',
                'Accuracy %': acc * 100, 
                'Recall %': rec * 100, 
                'F1 %': f1 * 100, 
                'Precision %': prec * 100
            })

        return pd.DataFrame(results).sort_values(by='Accuracy %', ascending=False)

    except Exception as e:
        st.error(f"An error occurred during model loading or prediction: {str(e)}")
        return pd.DataFrame()

# Streamlit UI setup
st.set_page_config(page_title="Fraud Detection Etherium Prediction App", page_icon="🕵️", layout="wide")
st.title("😎 **Fraud Detection Etherium Prediction App**")
st.subheader("Enter the following information to predict **Fraud Detection Etherium**.")

st.sidebar.title("🕵️ **Fraud Detection Parameters**")

# Input features
input_features = {
    "Avg min between sent tnx": st.sidebar.number_input("Avg min between sent tnx", min_value=0.0, value=float(df["Avg min between sent tnx"].mean())),
    "Avg min between received tnx": st.sidebar.number_input("Avg min between received tnx", min_value=0.0, value=float(df["Avg min between received tnx"].mean())),
    "Time difference between first and last (mins)": st.sidebar.number_input("Time difference between first and last (mins)", min_value=0.0, value=float(df["Time difference between first and last (mins)"].mean())),
    "Sent tnx": st.sidebar.number_input("Sent tnx", min_value=0.0, value=float(df["Sent tnx"].mean())),
    "Received tnx": st.sidebar.number_input("Received tnx", min_value=0.0, value=float(df["Received tnx"].mean())),
    "Number of created contracts": st.sidebar.number_input("Number of created contracts", min_value=0, value=int(df["Number of created contracts"].mean())),
    "Max value received": st.sidebar.number_input("Max value received", min_value=0.0, value=float(df["Max value received"].mean())),
    "Avg value received": st.sidebar.number_input("Avg value received", min_value=0.0, value=float(df["Avg value received"].mean())),
    "Avg value sent": st.sidebar.number_input("Avg value sent", min_value=0.0, value=float(df["Avg value sent"].mean())),
    "Total either sent": st.sidebar.number_input("Total either sent", min_value=0.0, value=float(df["Total either sent"].mean())),
    "Total either balance": st.sidebar.number_input("Total either balance", min_value=0.0, value=float(df["Total either balance"].mean())),
    "ERC20 total either received": st.sidebar.number_input("ERC20 total either received", min_value=0.0, value=float(df["ERC20 total either received"].mean())),
    "ERC20 total either sent": st.sidebar.number_input("ERC20 total either sent", min_value=0.0, value=float(df["ERC20 total either sent"].mean())),
    "ERC20 total either sent contract": st.sidebar.number_input("ERC20 total either sent contract", min_value=0.0, value=float(df["ERC20 total either sent contract"].mean())),
    "ERC20 unique sent address": st.sidebar.number_input("ERC20 unique sent address", min_value=0.0, value=float(df["ERC20 unique sent address"].mean())),
    "ERC20 unique received token name": st.sidebar.number_input("ERC20 unique received token name", min_value=0.0, value=float(df["ERC20 unique received token name"].mean())),
}

# Display predict button in main area
st.markdown("---")
if st.button(label=':rainbow[Predict Fraud]'):
    # Prepare input data for prediction
    input_data = pd.DataFrame([input_features])

    # Predicting the input data
    results_df = load_and_predict(input_data)

    # Displaying results
    if not results_df.empty:
        st.write("### 😎 Prediction Results:")
        styled_df = results_df.style.map(lambda x: 'color: green' if x == 'Yes' else 'color: red', subset=['Predicted Fraud'])
        st.dataframe(styled_df)

# Description Section
st.markdown("---")
st.subheader("Description")
st.markdown('''This Streamlit application predicts fraud in Ethereum transactions using multiple machine learning models including LGBM, XGBoost, and Gradient Boosting classifiers. 
Users can input transaction information through a user-friendly interface, which includes various fields related to transaction metrics and user activity.
> **Features:**
> - **Input Components:** Users can provide data using number inputs for transaction-related features.
> - **Data Processing:** Upon submitting the form, the app processes the input data and transforms it using a pre-trained data preprocessor. 
> - It leverages SMOTE to address any class imbalance in the data.
> - **Prediction:** The app runs predictions using the loaded models and calculates performance metrics like accuracy, recall, F1 score, and precision.
> - **Results Display:** The predicted fraud status and model performance metrics are displayed in a formatted output for easy interpretation.
> **Usage:** Just fill out the information about the transaction and click "Predict Fraud" to receive insights on whether the transaction is likely to be fraudulent and how well each model performed.
> **Disclaimer:** This application is intended for educational purposes only.
''')

# Disclaimer Section
st.markdown("---")
st.subheader("Disclaimer")
st.text('''The fraud detection results provided by this app are for informational purposes only. 
While we strive for accuracy, the predictions made by the models depend on the quality of the input data 
and the model's training. Use this information at your own discretion, and do not solely rely on it for 
making financial decisions. Consulting with a financial expert is recommended for critical decisions.''')
