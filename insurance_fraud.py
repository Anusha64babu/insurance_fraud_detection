import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Sample dataset
data = {
    "claim_amount": [50000, 150000, 10000, 80000],
    "policy_annual_premium": [20000, 40000, 15000, 30000],
    "customer_age": [35, 50, 42, 28],
    "num_previous_claims": [1, 4, 0, 2],
    "fraud_reported": ["N", "Y", "N", "Y"]
}
df = pd.DataFrame(data)

# Feature engineering
df["claim_to_policy_ratio"] = df["claim_amount"] / df["policy_annual_premium"]

X = df[["claim_amount","policy_annual_premium","customer_age","num_previous_claims","claim_to_policy_ratio"]]
y = (df["fraud_reported"] == "Y").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "fraud_detection_model.pkl")
print("✅ Fraud Detection model saved as fraud_detection_model.pkl")
