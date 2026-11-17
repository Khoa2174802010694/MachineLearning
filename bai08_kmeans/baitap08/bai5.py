'''
Bài tập 5: DBSCAN trên hai vầng trăng
Dùng DBSCAN(eps=e, min_samples=5) trong sklearn.cluster trên bộ hai vầng trăng,
với e lần lượt bằng 0.2, 0.3 và 0.4. Với mỗi giá trị, đếm số cụm tìm được và số điểm bị gán
nhãn -1, tức điểm nhiễu.
Có giá trị nào tách được hai vầng trăng tốt hơn K-Means không? Nhận xét vai trò của eps.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/hai_vang_trang.csv")

X = df[["x1", "x2"]].to_numpy()
nhan_that = df["nhan"].to_numpy()

km = KMeans(n_clusters=2, n_init=10, random_state=0)
nhan_kmeans = km.fit_predict(X)

ari_kmeans = adjusted_rand_score(nhan_that, nhan_kmeans)

print("KET QUA K-MEANS")
print("So cum:", len(set(nhan_kmeans)))
print("ARI:", round(ari_kmeans, 4))

ket_qua = []

for e in [0.2, 0.3, 0.4]:
    db = DBSCAN(eps=e, min_samples=5)
    nhan_dbscan = db.fit_predict(X)

    so_cum = len(set(nhan_dbscan) - {-1})
    so_nhieu = np.sum(nhan_dbscan == -1)
    ari = adjusted_rand_score(nhan_that, nhan_dbscan)

    ket_qua.append({
        "eps": e,
        "so_cum": so_cum,
        "so_diem_nhieu": so_nhieu,
        "ARI": ari
    })

    plt.figure(figsize=(7, 5))
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=nhan_dbscan,
        cmap="viridis",
        s=20
    )
    plt.title(
        f"DBSCAN (eps={e}, clusters={so_cum}, noise={so_nhieu})"
    )
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.tight_layout()
    plt.savefig(f"figures/bai5_dbscan_eps_{int(e * 10):02d}.png", dpi=150)
    plt.close()

ket_qua = pd.DataFrame(ket_qua)
ket_qua.to_csv("outputs/bai5_ket_qua.csv", index=False)

print("\nKET QUA DBSCAN")
print(ket_qua.to_string(index=False))

print("\nSO SANH VOI K-MEANS")
print("ARI K-Means:", round(ari_kmeans, 4))

for _, dong in ket_qua.iterrows():
    print(
        f"eps={dong['eps']}: "
        f"So cum={dong['so_cum']}, "
        f"So diem nhieu={dong['so_diem_nhieu']}, "
        f"ARI={dong['ARI']:.4f}"
    )
