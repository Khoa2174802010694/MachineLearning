'''
Bài tập 6: Phân cụm chữ số viết tay
Chạy K-Means với K = 10 trên bộ chữ số viết tay load_digits, không dùng nhãn. Sau
đó mới dùng nhãn thật: lập bảng pd.crosstab(nhan_that, cum) để xem mỗi cụm chứa
những chữ số nào.
Chữ số nào được gom gọn vào một cụm? Những chữ số nào hay bị gom chung?
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

digits = load_digits()

X = digits.data
nhan_that = digits.target

km = KMeans(
    n_clusters=10,
    n_init=10,
    random_state=0
)

cum = km.fit_predict(X)

bang = pd.crosstab(
    nhan_that,
    cum,
    rownames=["Nhan that"],
    colnames=["Cum"]
)

bang.to_csv("outputs/bai6_bang_doi_chieu.csv")

ari = adjusted_rand_score(nhan_that, cum)

print("KET QUA PHAN CUM CHU SO")
print("So mau:", len(X))
print("So cum:", len(np.unique(cum)))
print("ARI:", round(ari, 4))

print("\nBANG DOI CHIEU NHAN THAT VA CUM")
print(bang)

print("\nCHU SO CHIEM UU THE TRONG TUNG CUM")

for c in bang.columns:
    chu_so = bang[c].idxmax()
    so_luong = bang[c].max()
    tong = bang[c].sum()

    print(
        f"Cum {c}: Chu so {chu_so} "
        f"chiem uu the voi {so_luong}/{tong} mau"
    )

plt.figure(figsize=(10, 7))
plt.imshow(bang, cmap="Blues", aspect="auto")
plt.colorbar(label="So luong mau")
plt.xticks(
    range(len(bang.columns)),
    bang.columns
)
plt.yticks(
    range(len(bang.index)),
    bang.index
)
plt.xlabel("Cum K-Means")
plt.ylabel("Nhan that")
plt.title("Doi chieu nhan that va ket qua phan cum")
plt.tight_layout()
plt.savefig("figures/bai6_chu_so_heatmap.png", dpi=150)
plt.close()

with open("outputs/bai6_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write("KET QUA PHAN CUM CHU SO\n")
    f.write(f"So mau: {len(X)}\n")
    f.write(f"So cum: {len(np.unique(cum))}\n")
    f.write(f"ARI: {ari:.4f}\n\n")

    f.write("BANG DOI CHIEU\n")
    f.write(bang.to_string())
    f.write("\n\nCHU SO CHIEM UU THE TRONG TUNG CUM\n")

    for c in bang.columns:
        chu_so = bang[c].idxmax()
        so_luong = bang[c].max()
        tong = bang[c].sum()

        f.write(
            f"\nCum {c}: Chu so {chu_so} "
            f"chiem uu the voi {so_luong}/{tong} mau"
        )

print("\nDa luu bang doi chieu va bieu do.")
