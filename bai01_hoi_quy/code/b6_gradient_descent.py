import numpy as np
import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Chuẩn hóa x
x_tb = x_goc.mean()
x_std = x_goc.std()
x = (x_goc - x_tb) / x_std

w, b = 0.0, 0.0
learning_rate = 0.1
n = len(x)

for i in range(200):
    y_pred = w * x + b
    error = y_pred - y

    grad_w = (2 / n) * (error * x).sum()
    grad_b = (2 / n) * error.sum()

    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

    if i + 1 in [1, 2, 5, 10, 25, 50, 100, 200]:
        mse = ((w * x + b - y) ** 2).mean()
        print(f"Vong {i+1:3d}: w={w:.4f}, b={b:.4f}, MSE={mse:.4f}")

# Đưa w, b về đơn vị ban đầu
w_goc = w / x_std
b_goc = b - w * x_tb / x_std

print(f"\nw = {w_goc:.6f}")
print(f"b = {b_goc:.6f}")