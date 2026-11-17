'''
Bài tập 2: Thêm cột tuổi
Phân cụm trên cả ba cột tuổi, thu nhập và điểm chi tiêu, có chuẩn hoá. Chọn lại K bằng
silhouette với K từ 2 tới 10.
K tốt nhất có còn là 5 không? Tâm các cụm thay đổi thế nào so với mục 3?
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/khach_hang.csv")

cot = ["tuoi", "thu_nhap", "diem_chi_tieu"]
X = df[cot]

sc = StandardScaler()
Xs = sc.fit_transform(X)

ket_qua = []

for k in range(2, 11):
    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=0
    )

    nhan = km.fit_predict(Xs)
    sil = silhouette_score(Xs, nhan)

    ket_qua.append({
        "K": k,
        "Inertia": km.inertia_,
        "Silhouette": sil
    })

bang = pd.DataFrame(ket_qua)
bang.to_csv("outputs/bai2_ket_qua.csv", index=False)

i_tot = bang["Silhouette"].idxmax()
k_tot = int(bang.loc[i_tot, "K"])

km_moi = KMeans(
    n_clusters=k_tot,
    n_init=10,
    random_state=0
)

nhan_moi = km_moi.fit_predict(Xs)

tam_moi = pd.DataFrame(
    sc.inverse_transform(km_moi.cluster_centers_),
    columns=cot
)

tam_moi["so_khach"] = (
    pd.Series(nhan_moi)
    .value_counts()
    .sort_index()
    .values
)

tam_moi.to_csv("outputs/bai2_tam_cum.csv", index=False)

print("KET QUA SILHOUETTE THEO K")
print(bang.to_string(index=False))
print()
print(f"K tot nhat: {k_tot}")
print(f"Silhouette tot nhat: {bang.loc[i_tot, 'Silhouette']:.4f}")
print()
print("Tam cum moi:")
print(tam_moi.round(2).to_string(index=False))

plt.figure(figsize=(8, 5))

plt.plot(
    bang["K"],
    bang["Silhouette"],
    marker="o"
)

plt.xticks(range(2, 11))
plt.xlabel("So cum K")
plt.ylabel("Silhouette")
plt.title("Chon K bang Silhouette voi ba thuoc tinh")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/bai2_silhouette.png", dpi=200)
plt.show()
