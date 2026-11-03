import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

plt.scatter(df["so_phong"], df["gia"])

plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")
plt.title("Moi quan he giua so phong va gia")

plt.savefig("bai2.png", dpi=150)
plt.show()

print("Nhan xet: So phong tang thi gia nha co xu huong tang.")