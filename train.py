import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from src.features import AirbnbFeatureBuilder

# 1️⃣ Load data
df = pd.read_csv("data/airbnb_cleaned.csv")

# 2️⃣ Features & target
target = "price"
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# 3️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Pipeline
pipe = Pipeline([
    ("features", AirbnbFeatureBuilder()),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42))
])

# 5️⃣ Train
pipe.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = pipe.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ RMSE: {rmse:.2f} | R²: {r2:.3f}")

# 7️⃣ Save model
joblib.dump(pipe, "models/airbnb_pipeline.pkl")
print("💾 Model saved to models/airbnb_pipeline.pkl")
