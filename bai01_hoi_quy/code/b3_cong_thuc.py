import numpy as np
import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

x = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()

w = ((x - x_tb) * (y - y_tb)).sum() / ((x - x_tb) ** 2).sum()
b = y_tb - w * x_tb

print(f"x trung binh = {x_tb:.4f}")
print(f"y trung binh = {y_tb:.4f}")
print(f"w = {w:.6f}")
print(f"b = {b:.6f}")

print(f"\nMo hinh: gia = {w:.4f} * dien_tich + {b:.4f}")

y_du_doan = w * x + b
mse = ((y - y_du_doan) ** 2).mean()

print(f"MSE = {mse:.4f}")
print(f"Du doan nha 80 m2: {w * 80 + b:.3f} ty")