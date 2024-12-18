import os
import joblib
import pandas as pd
from typing import Any, Dict, List
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr

# Constants for directories and file names
MODEL_DIR = 'models'
DATA_DIR = 'datasets'
DATA_FILE = 'cleaned_transaction_dataset.csv'
MODEL_NAMES = [
    'LGBM Classifier', 
    'CatBoost Classifier',
    'XGBoost Classifier', 
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
            print(f"Error loading model {name}: {str(e)}")
    return models

models = load_models(MODEL_NAMES)

# Prepare features and target
X = df.drop(columns=['FLAG'])
y = df['FLAG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

# Standardize the features
scaler = StandardScaler().fit(X_train)

# Prediction and metrics evaluation function
def calculate_metrics(y_true, y_pred, average_type='binary'):
    """Calculate and return accuracy, recall, F1, and precision scores."""
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, average=average_type)
    f1 = f1_score(y_true, y_pred, average=average_type)
    prec = precision_score(y_true, y_pred, average=average_type)
    return acc, rec, f1, prec

def load_and_predict(input_data):
    try:
        # Scale the input sample using the already-fitted scaler
        sample_trans = scaler.transform(input_data)

        # Using SMOTE to handle class imbalance
        X_resampled, y_resampled = SMOTE(random_state=123).fit_resample(scaler.transform(X_train), y_train)

        results = []

        for name, model in models.items():
            flag_pred = model.predict(sample_trans)
            y_resampled_pred = model.predict(X_resampled)
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
        return f"An error occurred during prediction: {str(e)}"

# Gradio interface
def predict(avg_min_sent, avg_min_received, time_diff, sent_tnx, received_tnx, num_created_contracts,
            max_value_received, avg_value_received, avg_value_sent, total_sent,
            total_balance, erc20_received, erc20_sent, erc20_sent_contract,
            erc20_unique_sent, erc20_unique_received):

    input_features = [
        avg_min_sent,
        avg_min_received,
        time_diff,
        sent_tnx,
        received_tnx,
        num_created_contracts,
        max_value_received,
        avg_value_received,
        avg_value_sent,
        total_sent,
        total_balance,
        erc20_received,
        erc20_sent,
        erc20_sent_contract,
        erc20_unique_sent,
        erc20_unique_received
    ]
    
    input_data = pd.DataFrame([input_features])
    results_df = load_and_predict(input_data)

    return results_df

# Gradio inputs based on the features you have
inputs = [
    gr.Number(label="Avg min between sent tnx", value=df["Avg min between sent tnx"].mean()),
    gr.Number(label="Avg min between received tnx", value=df["Avg min between received tnx"].mean()),
    gr.Number(label="Time difference between first and last (mins)", value=df["Time difference between first and last (mins)"].mean()),
    gr.Number(label="Sent tnx", value=df["Sent tnx"].mean()),
    gr.Number(label="Received tnx", value=df["Received tnx"].mean()),
    gr.Number(label="Number of created contracts", value=int(df["Number of created contracts"].mean())),
    gr.Number(label="Max value received", value=df["Max value received"].mean()),
    gr.Number(label="Avg value received", value=df["Avg value received"].mean()),
    gr.Number(label="Avg value sent", value=df["Avg value sent"].mean()),
    gr.Number(label="Total either sent", value=df["Total either sent"].mean()),
    gr.Number(label="Total either balance", value=df["Total either balance"].mean()),
    gr.Number(label="ERC20 total either received", value=df["ERC20 total either received"].mean()),
    gr.Number(label="ERC20 total either sent", value=df["ERC20 total either sent"].mean()),
    gr.Number(label="ERC20 total either sent contract", value=df["ERC20 total either sent contract"].mean()),
    gr.Number(label="ERC20 unique sent address", value=df["ERC20 unique sent address"].mean()),
    gr.Number(label="ERC20 unique received token name", value=df["ERC20 unique received token name"].mean()),
]

output = gr.Dataframe(label="Prediction Results")

# Create the Gradio interface
gr.Interface(
    fn=predict, 
    inputs=inputs, 
    outputs=output,
    title="Fraud Detection Etherium Prediction App",
    description="This application predicts fraud in Ethereum transactions using multiple machine learning models.",
    theme="compact"
).launch()
