import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

print("Kich thuoc:", df.shape)
print("\n5 dong dau:")
print(df.head())

print("\nTen cot:", list(df.columns))

print("\nThong ke:")
print(df[["dien_tich", "gia"]].describe().round(2))

print("\nGia tri thieu:")
print(df.isna().sum())