import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

model = LinearRegression()
model.fit(X, y)

print(f"w = {model.coef_[0]:.6f}")
print(f"b = {model.intercept_:.6f}")

print("\nSo voi buoc 3:")
print("w = 0.078367, b = 0.401752")

for dt in [80, 100]:
    gia = model.predict([[dt]])[0]
    print(f"Nha {dt} m2 -> {gia:.3f} ty dong")