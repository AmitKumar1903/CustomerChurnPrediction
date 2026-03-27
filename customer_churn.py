import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import joblib

# Load data
data = pd.read_csv("C:\\Users\\amitt\\Downloads\\churn.csv")

# Data cleaning
data = data.drop(columns=['CustomerId', 'RowNumber', 'Surname'], axis=1)
data = data.dropna()

# Feature engineering
data['BalanceToSalaryRatio'] = np.log1p(data['Balance']) / (data['EstimatedSalary'] + 1e-6)
data['HasHighBalance'] = (data['Balance'] > data['EstimatedSalary']).astype(int)

# Create categorical features
data['Tenure_Bucket'] = pd.cut(
    data['Tenure'],
    bins=[0, 3, 7, 100],
    labels=['0-3', '4-7', '8+']
)

data['CreditScore_Bucket'] = pd.cut(
    data['CreditScore'],
    bins=[0, 500, 650, 800, 1000],
    labels=['Poor', 'Average', 'Good', 'Excellent']
)

# Encode categorical variables
categorical_cols = ['Geography', 'Gender', 'Tenure_Bucket', 'CreditScore_Bucket']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Prepare features and target
X = data.drop('Exited', axis=1)
y = data['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
    shuffle=True
)

# Feature scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote_enn = SMOTEENN(random_state=42)
X_res, y_res = smote_enn.fit_resample(X_train_scaled, y_train)

# Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.7, 1.0],
    'scale_pos_weight': [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]
}

xgb = XGBClassifier(
    n_estimators=100,
    random_state=42,
    eval_metric='logloss'
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_res, y_res)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print("Best Parameters:", grid_search.best_params_)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nKey Metrics:")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")

# Feature importance
feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)

plt.figure(figsize=(10, 6))
feature_importance.nlargest(15).plot(kind='barh')
plt.title('Top 15 Feature Importances')
plt.show()

# Save model
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")