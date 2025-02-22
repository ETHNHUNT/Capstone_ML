import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#Select relevant features
features = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
df = df[features + ["Survived"]]

# Handle missing values
df = df.assign(Age=df["Age"].fillna(df["Age"].median()))
df = df.assign(Embarked=df["Embarked"].fillna("S"))

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Train-Test Split
X = df.drop(columns=["Survived"])
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# Train XGBoost Model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluate Models
def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }

rf_metrics = evaluate_model(y_test, rf_preds)
xgb_metrics = evaluate_model(y_test, xgb_preds)

# Compare Models
print("Random Forest Metrics:", rf_metrics)
print("XGBoost Metrics:", xgb_metrics)