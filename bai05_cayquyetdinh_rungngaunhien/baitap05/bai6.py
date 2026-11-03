from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gia_nha.csv"

df = pd.read_csv(DATA_PATH)
X = df[["dien_tich"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=0),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=0)
}

for ten, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{ten} MAE: {mae:.4f}")

dien_tich_moi = pd.DataFrame({"dien_tich": [250]})

print("\nDu doan gia nha voi dien tich 250 m2:")
for ten, model in models.items():
    gia_du_doan = model.predict(dien_tich_moi)[0]
    print(f"{ten}: {gia_du_doan:.4f}")

rf = models["Random Forest"]
gia_min_train = y_train.min()
gia_max_train = y_train.max()
gia_rf = rf.predict(dien_tich_moi)[0]

print("\nKhoang gia trong tap train:", gia_min_train, "->", gia_max_train)
print("Gia du doan Random Forest:", gia_rf)
print(
    "Random Forest khong ngoai suy tuyen tinh ra ngoai mien du lieu. "
    "Moi cay du doan bang gia trung binh cua cac mau train trong la, "
    "sau do rung lay trung binh cac cay."
)
