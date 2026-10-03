import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

# Chia 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Train: {len(X_train)} can")
print(f"Test : {len(X_test)} can")

print(f"\nMAE  = {mae:.4f} ty")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f} ty")
print(f"R2   = {r2:.4f}")

for dt, that, pred in zip(
    X_test["dien_tich"], y_test, y_pred
):
    print(f"{dt:.1f} m2 | that={that:.2f} | du doan={pred:.2f}")