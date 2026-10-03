import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

nhom_lon = df[df["dien_tich"] > 100]

print("So can tren 100 m2:", len(nhom_lon))
print("Gia trung binh:", nhom_lon["gia"].mean(), "ty")