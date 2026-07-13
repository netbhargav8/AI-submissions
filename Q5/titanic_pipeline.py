import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# 1. Dataset Verification & Fallback Setup
# ==========================================
if not os.path.exists('titanic.csv'):
    print("No 'titanic.csv' detected. Creating synthetic Titanic dataset...")
    np.random.seed(1)
    rows = 200
    mock_data = {
        'Survived': np.random.choice([0, 1], size=rows, p=[0.6, 0.4]),
        'Pclass': np.random.choice([1, 2, 3], size=rows, p=[0.25, 0.25, 0.50]),
        'Sex': np.random.choice(['male', 'female'], size=rows, p=[0.60, 0.40]),
        'Age': np.random.choice([np.nan, 22, 38, 26, 35, 54, 2, 27, 14, 4], size=rows),
        'SibSp': np.random.choice([0, 1, 2, 3], size=rows, p=[0.7, 0.2, 0.05, 0.05]),
        'Parch': np.random.choice([0, 1, 2], size=rows, p=[0.8, 0.1, 0.1]),
        'Fare': np.round(np.random.uniform(7.25, 150.0, size=rows), 2)
    }
    pd.DataFrame(mock_data).to_csv('titanic.csv', index=False)

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
print("Executing Step 1: Performing Exploratory Data Analysis...")
df = pd.read_csv('titanic.csv')
df.columns = df.columns.str.strip()

print(f"\n--- Baseline Dataset Shape: {df.shape[0]} Passengers, {df.shape[1]} Features ---")
print("\n[EDA] Survival Rate Profile by Gender:")
print(df.groupby('Sex')['Survived'].mean().to_string())

print("\n[EDA] Survival Rate Profile by Passenger Class (Pclass):")
print(df.groupby('Pclass')['Survived'].mean().to_string())

# ==========================================
# 3. Data Cleansing & Feature Engineering
# ==========================================
print("\nExecuting Step 2: Cleansing data & engineering new features...")

# Feature Engineering: Combine individual metrics into Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Isolate Features (X) and Target (y)
X = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']].copy()
y = df['Survived'].copy()

# Encoding text categorical vectors (Sex: male -> 1, female -> 0)
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])

# Machine Learning Imputation: Patch missing Age cells using the Median value
imputer = SimpleImputer(strategy='median')
X['Age'] = imputer.fit_transform(X[['Age']])

# Perform standard 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ==========================================
# 4. Model Duel Training & Evaluation
# ==========================================
print("Executing Step 3: Running Model Duel (Logistic Regression vs Random Forest)...")

# Initialize models
log_reg = LogisticRegression(max_iter=500, random_state=1)
rand_forest = RandomForestClassifier(n_estimators=100, random_state=1)

# Fit models
log_reg.fit(X_train, y_train)
rand_forest.fit(X_train, y_train)

# Generate predictions
y_pred_lr = log_reg.predict(X_test)
y_pred_rf = rand_forest.predict(X_test)

# Helper function to compile performance matrix
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred) * 100,
        precision_score(y_true, y_pred) * 100,
        recall_score(y_true, y_pred) * 100
    ]

lr_metrics = get_metrics(y_test, y_pred_lr)
rf_metrics = get_metrics(y_test, y_pred_rf)

# ==========================================
# 5. Outputting Performance Diagnostics
# ==========================================
print("\n" + "="*55)
print("      COMPETING CLASSIFICATION PERFORMANCE MATRIX     ")
print("="*55)
print(f"METRIC       | LOGISTIC REGRESSION | RANDOM FOREST   ")
print(f"-------------------------------------------------------")
print(f"Accuracy     | {lr_metrics[0]:.2f}%               | {rf_metrics[0]:.2f}%")
print(f"Precision    | {lr_metrics[1]:.2f}%               | {rf_metrics[1]:.2f}%")
print(f"Recall       | {lr_metrics[2]:.2f}%               | {rf_metrics[2]:.2f}%")
print("="*55 + "\n")