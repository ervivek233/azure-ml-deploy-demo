import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib

# Load large dataset
df = pd.read_csv("data/sensor_data_large.csv")

X = df.drop("failure", axis=1)
y = df["failure"]

# Stratified split to handle imbalance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = XGBClassifier(
    scale_pos_weight=5,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation (safe metrics)
print(
    classification_report(
        y_test,
        model.predict(X_test),
        zero_division=0
    )
)

# Save model
joblib.dump(model, "model.pkl")

print("Model training completed and saved as model.pkl")
