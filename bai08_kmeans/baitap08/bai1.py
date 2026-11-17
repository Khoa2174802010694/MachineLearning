'''
Bài tập 1: Tự tính silhouette
Viết hàm silhouette_mot_diem(X, nhan, i) tính a, b và s của điểm thứ i theo đúng định
nghĩa ở mục 4, chỉ dùng numpy. Chạy hàm cho mọi khách hàng với K = 5, lấy trung bình,
rồi so với silhouette_score.
Khách nào có s nhỏ nhất? Khách đó nằm ở đâu trên Hình 3?
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def silhouette_mot_diem(X, nhan, i):
    khoang_cach = np.sqrt(((X - X[i]) ** 2).sum(axis=1))

    cung_cum = nhan == nhan[i]
    cung_cum[i] = False

    if not cung_cum.any():
        return 0.0, 0.0, 0.0

    a = khoang_cach[cung_cum].mean()

    cac_cum_khac = np.unique(nhan[nhan != nhan[i]])

    b = min(
        khoang_cach[nhan == cum].mean()
        for cum in cac_cum_khac
    )

    s = (b - a) / max(a, b)

    return a, b, s

df = pd.read_csv("data/khach_hang.csv")

X_goc = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X_goc)

km = KMeans(
    n_clusters=5,
    n_init=10,
    random_state=0
)

nhan = km.fit_predict(Xs)

ket_qua = np.array([
    silhouette_mot_diem(Xs, nhan, i)
    for i in range(len(Xs))
])

sil_numpy = ket_qua[:, 2].mean()
sil_sklearn = silhouette_score(Xs, nhan)

i_min = int(np.argmin(ket_qua[:, 2]))

with open("outputs/bai1_ket_qua.txt", "w", encoding="utf-8") as f:
    f.write("BAI 1: TU TINH SILHOUETTE\n")
    f.write(f"Silhouette NumPy: {sil_numpy:.6f}\n")
    f.write(f"Silhouette sklearn: {sil_sklearn:.6f}\n")
    f.write(f"Do chenh lech: {abs(sil_numpy - sil_sklearn):.10f}\n\n")
    f.write(f"Khach hang co Silhouette nho nhat: {i_min + 1}\n")
    f.write(f"Thu nhap: {X_goc.iloc[i_min, 0]}\n")
    f.write(f"Diem chi tieu: {X_goc.iloc[i_min, 1]}\n")
    f.write(f"Cum: {nhan[i_min]}\n")
    f.write(f"a = {ket_qua[i_min, 0]:.4f}\n")
    f.write(f"b = {ket_qua[i_min, 1]:.4f}\n")
    f.write(f"s = {ket_qua[i_min, 2]:.4f}\n")

print("BAI 1: TU TINH SILHOUETTE")
print(f"Silhouette NumPy: {sil_numpy:.6f}")
print(f"Silhouette sklearn: {sil_sklearn:.6f}")
print(f"Do chenh lech: {abs(sil_numpy - sil_sklearn):.10f}")
print()
print("Khach hang co Silhouette nho nhat:")
print(f"Vi tri: {i_min + 1}")
print(f"Thu nhap: {X_goc.iloc[i_min, 0]}")
print(f"Diem chi tieu: {X_goc.iloc[i_min, 1]}")
print(f"Cum: {nhan[i_min]}")
print(f"a = {ket_qua[i_min, 0]:.4f}")
print(f"b = {ket_qua[i_min, 1]:.4f}")
print(f"s = {ket_qua[i_min, 2]:.4f}")

plt.figure(figsize=(9, 6))

plt.scatter(
    X_goc["thu_nhap"],
    X_goc["diem_chi_tieu"],
    c=nhan,
    cmap="viridis",
    s=35,
    alpha=0.7
)

plt.scatter(
    X_goc.iloc[i_min, 0],
    X_goc.iloc[i_min, 1],
    color="red",
    marker="X",
    s=200,
    edgecolors="black",
    label="Silhouette thap nhat"
)

plt.annotate(
    f"Khach {i_min + 1}",
    (X_goc.iloc[i_min, 0], X_goc.iloc[i_min, 1]),
    xytext=(10, 10),
    textcoords="offset points"
)

plt.xlabel("Thu nhap")
plt.ylabel("Diem chi tieu")
plt.title("K-Means K=5 va khach hang co Silhouette thap nhat")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("figures/bai1_silhouette.png", dpi=200)
plt.show()
