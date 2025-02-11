# Customer Churn Prediction 📊🚀

## Overview
Welcome to the **Customer Churn Prediction** project! 🎯 This machine learning model predicts whether a customer is likely to churn based on their demographic, account, and transaction data. By using customer insights, we can proactively take action to retain valuable clients.

The web app, built using **Streamlit**, allows users to input customer data and get a real-time churn prediction. This project demonstrates how machine learning can be applied to solve real-world business challenges. 📉💡
**Live Demo**: Check out the live version of the app here: [Customer Churn Prediction Web App](https://customerchurnprediction-a.streamlit.app/) 🌐

## 🗂️ Project Structure
├── model_training.py # Model training, data preprocessing, feature engineering, and hyperparameter tuning. ├── app.py # Streamlit web app for customer churn prediction. ├── model_churn.pkl # The trained machine learning model. ├── scaler.pkl # Scaler used for feature normalization. └── churn.csv # The dataset containing customer data.


## 📊 Dataset
The dataset `churn.csv` contains vital customer data used for churn prediction. The key features include:

- **Customer details**: Credit score, age, tenure, geography, gender.
- **Banking details**: Account balance, number of products, credit card ownership, activity status.
- **Target variable**: `Exited` (1 = Churned, 0 = Not Churned).

## 🚀 Model Training Pipeline
### 1. **Data Preprocessing**
   - Removed unnecessary columns (`CustomerId`, `RowNumber`, `Surname`).
   - Handled missing values and encoded categorical variables.
   - Applied **RobustScaler** for feature normalization.
   - Used **SMOTEENN** for handling class imbalance.

### 2. **Feature Engineering**
   - Created derived features: `BalanceToSalaryRatio`, `HasHighBalance`, `Tenure_Bucket`, and `CreditScore_Bucket`.

### 3. **Model Selection & Hyperparameter Tuning**
   - Algorithm: **XGBoost Classifier**.
   - Hyperparameters optimized using **GridSearchCV**.
   - Key metrics: **F1-score**, **ROC-AUC**, **Recall**, and **Feature Importance**.

### 4. **Model Evaluation**
   - The model achieved a high ROC-AUC score, ensuring strong predictive power.
   - Feature importance showed that **Age**, **Balance**, and **Number of Products** were the top predictors of churn.

## 🌐 Web Application (Streamlit)
This project includes a user-friendly web application where users can input customer details and receive a churn prediction. Here’s what the app offers:

- **User Inputs**: Age, credit score, geography, balance, etc.
- **Preprocessing**: Input data is transformed to match the model’s feature set.
- **Prediction**: The trained model predicts whether the customer is likely to churn.
- **Visualization**: Display of feature importance and data distribution.
This project allows users to predict customer churn in two ways:
✅ Single Entry Mode – Users can manually input customer details and get an instant prediction.
✅ Bulk Prediction Mode – Users can upload a CSV file, and the app will process multiple entries at once, returning a CSV file with predictions.

📌 How to Use the Bulk Prediction Feature
Prepare a CSV file with customer details (must match the required input format).
Upload the file via the web app’s CSV upload feature.
The app processes the file and applies the trained model to make predictions.
Download the output CSV containing churn predictions for all customers.

## ⚙️ How to Run the Project
To run this project locally, follow these simple steps:

### 1. Install Dependencies
Ensure you have the required dependencies by running the following command:
```bash
pip install pandas numpy joblib streamlit scikit-learn xgboost seaborn imbalanced-learn matplotlib
```
### 2. Train the Model
Run model_training.py to preprocess the data, train the model, and save the trained model and scaler:
```bash
python model_training.py
```
### 3. Run the Web App
Launch the Streamlit web app with this command:
```bash
streamlit run app.py
```

## 🏆 Results & Insights
- **Feature Importance**: Age, balance, and the number of products were identified as the most significant factors contributing to churn.
- **Performance**: The tuned XGBoost model performed exceptionally well with a high ROC-AUC score, validating its predictive strength.

## 🤝 Contributing
Contributions are always welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

## 📑 License
This project is licensed under the MIT License - see the LICENSE file for details.
