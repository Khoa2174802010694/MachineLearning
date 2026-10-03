import numpy as np
import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()

w = ((x - x_tb) * (y - y_tb)).sum() / ((x - x_tb) ** 2).sum()
b = y_tb - w * x_tb

print(f"w = {w:.6f}")
print(f"b = {b:.6f}")

print("Nhan xet: w mang dau am, vi tuoi nha tang thi gia nha co xu huong giam.")