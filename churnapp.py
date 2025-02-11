import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler


def load_model():
    model = joblib.load("model_churn.pkl")
    scaler = joblib.load("scaler_churn.pkl")
    return model, scaler


# Load the trained model and scaler
model, scaler = load_model()

# Get the expected feature names from the trained model
expected_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
    'IsActiveMember', 'EstimatedSalary', 'BalanceToSalaryRatio', 'HasHighBalance', 'Geography_Germany', 'Geography_Spain', 'Gender_Male', 
    'Tenure_Bucket_4-7', 'Tenure_Bucket_8+', 
    'CreditScore_Bucket_Average', 'CreditScore_Bucket_Good', 'CreditScore_Bucket_Excellent'
]

# Function to preprocess input data
def preprocess_data(data):
    # Feature Engineering
    data['BalanceToSalaryRatio'] = np.log1p(data['Balance']) / (data['EstimatedSalary'] + 1e-6)
    data['HasHighBalance'] = (data['Balance'] > data['EstimatedSalary']).astype(int)

    # Categorize and Encode
    data['Tenure_Bucket'] = pd.cut(data['Tenure'], bins=[0, 3, 7, 100], labels=['0-3', '4-7', '8+'])
    data['CreditScore_Bucket'] = pd.cut(data['CreditScore'], bins=[0, 500, 650, 800, 1000],
                                        labels=['Poor', 'Average', 'Good', 'Excellent'])
    categorical_cols = ['Geography', 'Gender', 'Tenure_Bucket', 'CreditScore_Bucket']
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Ensure input matches training features
    data = data.reindex(columns=expected_features, fill_value=0)

    # Scaling (if used during training)
    data_scaled = scaler.transform(data)

    return data_scaled


# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# **App Header Section**
st.title("Customer Churn Prediction")
st.markdown("""
    This app predicts whether a customer is likely to churn based on their account details.
    Please fill in the details below to get a prediction.
""")

# **User Input Section (Side-by-Side Inputs using st.columns)**
col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
    balance = st.number_input("Account Balance", min_value=0.0, max_value=250000.0, value=50000.0)
    has_card = st.selectbox("Has Credit Card?", [0, 1])

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    is_active = st.selectbox("Is Active Member?", [0, 1])
    salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0)

# **Prediction Button**
if st.button("Predict Churn", use_container_width=True):
    # Convert input to DataFrame
    input_data = pd.DataFrame([{
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': products,
        'HasCrCard': has_card,
        'IsActiveMember': is_active,
        'EstimatedSalary': salary
    }])

    # Preprocess and predict
    processed_input = preprocess_data(input_data)
    prediction = model.predict(processed_input)
    churn_result = "🔴 Likely to Churn" if prediction[0] == 1 else "🟢 Not Likely to Churn"

    # Display result
    st.subheader(f"Prediction: {churn_result}")

# **CSV Upload Section**
st.header("📌 Upload CSV for Bulk Prediction")

# Display file uploader with more explanation
uploaded_file = st.file_uploader("Upload a CSV file with customer data for bulk churn prediction", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # **Handle missing or extra columns**
    missing_cols = set(expected_features) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_features)

    if missing_cols:
        st.warning(f"⚠️ Missing columns in CSV: {missing_cols}")

    if extra_cols:
        st.info(f"ℹ️ Extra columns in CSV (will be ignored): {extra_cols}")

    # Preprocess and predict
    processed_df = preprocess_data(df)
    predictions = model.predict(processed_df)
    df['Churn Prediction'] = predictions

    # Show and download results
    st.write("Prediction Results:")
    st.dataframe(df[['CustomerId', 'Churn Prediction']])
    st.download_button("Download Predictions", df.to_csv(index=False), "churn_predictions.csv")

# **Footer Section for App Launch**
st.markdown("""
    ---  
    *The Customer Churn Prediction app is powered by Streamlit and Scikit-Learn.*
""")
