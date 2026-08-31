from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
DATA_PATH = BASE_DIR / "data" / "AmesHousing.csv"

# Load model
model = joblib.load(MODEL_PATH)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Remove identifiers and target
house = df.drop(columns=["Order", "PID", "SalePrice"]).iloc[[0]]

# Predict
prediction = model.predict(house)[0]

print(f"Predicted house price: ${prediction:,.2f}")