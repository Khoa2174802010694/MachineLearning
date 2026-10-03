import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")
data = df.iloc[[0, 14, 29, 44, 59]]

x = data["dien_tich"].to_numpy()
y = data["gia"].to_numpy()


def mse(w, b):
    return ((y - (w * x + b)) ** 2).mean()


print("5 can nha:")
for xi, yi in zip(x, y):
    print(f"{xi:6.1f} m2 -> {yi:5.2f} ty")

for w, b in [(0.05, 1.5), (0.08, 0.5)]:
    print(f"\ny = {w}x + {b}")

    for xi, yi in zip(x, y):
        du_doan = w * xi + b
        sai_so = yi - du_doan
        print(f"x={xi:6.1f} | that={yi:5.2f} | du doan={du_doan:5.2f} | sai so={sai_so:+.2f}")

    print(f"MSE = {mse(w, b):.4f}")