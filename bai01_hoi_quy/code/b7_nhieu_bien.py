import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

y = df["gia"]
models = {
    "1 bien": ["dien_tich"],
    "3 bien": ["dien_tich", "so_phong", "tuoi_nha"]
}

for ten, cols in models.items():
    X = df[cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{ten}: R2 = {r2:.4f}")

# Xem hệ số của mô hình 3 biến
model = LinearRegression()
model.fit(df[models["3 bien"]], y)

print("\nHe so:")
for ten, coef in zip(models["3 bien"], model.coef_):
    print(f"{ten}: {coef:+.4f}")

print(f"intercept: {model.intercept_:+.4f}")