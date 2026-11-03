import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

X = df[["dien_tich", "so_phong"]]
y = df["gia"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R2 = {r2:.4f}")
print("So sanh: Mo hinh mot bien co R2 = 0.9622.")