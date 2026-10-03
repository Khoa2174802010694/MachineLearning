import numpy as np
import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x_goc.mean()
x_std = x_goc.std()
x = (x_goc - x_tb) / x_std

for toc_do_hoc in [0.001, 1.02]:
    w = 0.0
    b = 0.0
    n = len(x)

    for i in range(200):
        y_pred = w * x + b
        error = y_pred - y

        grad_w = (2 / n) * (error * x).sum()
        grad_b = (2 / n) * error.sum()

        w -= toc_do_hoc * grad_w
        b -= toc_do_hoc * grad_b

    mse = ((w * x + b - y) ** 2).mean()

    print(f"Tuoc do hoc = {toc_do_hoc}")
    print(f"MSE vong 200 = {mse:.4f}")
    print()