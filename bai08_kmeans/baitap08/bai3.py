'''
Bài tập 3: K-Means++ và khởi tạo ngẫu nhiên
Chạy K-Means 50 lần với n_init=1, random_state từ 0 tới 49, lần lượt với init="random"
và init="k-means++". Đếm xem mỗi cách có bao nhiêu lần đạt inertia 23.0. Cách khởi
tạo nào đáng tin hơn?
'''
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/khach_hang.csv")

X = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X)

ket_qua = []

for init in ["random", "k-means++"]:
    for rs in range(50):
        km = KMeans(
            n_clusters=5,
            init=init,
            n_init=1,
            random_state=rs
        )

        km.fit(Xs)

        ket_qua.append({
            "init": init,
            "random_state": rs,
            "inertia": km.inertia_,
            "inertia_lam_tron": round(km.inertia_, 1)
        })

bang = pd.DataFrame(ket_qua)

bang.to_csv(
    "outputs/bai3_ket_qua_100_lan.csv",
    index=False
)

tong_hop = bang.groupby("init").agg(
    so_lan_dat_23=(
        "inertia_lam_tron",
        lambda x: (x == 23.0).sum()
    ),
    inertia_min=("inertia", "min"),
    inertia_max=("inertia", "max"),
    inertia_trung_binh=("inertia", "mean")
).reset_index()

tong_hop.to_csv(
    "outputs/bai3_tong_hop.csv",
    index=False
)

print("KET QUA 50 LAN KHOI TAO")
print(tong_hop.round(3).to_string(index=False))

with open(
    "outputs/bai3_ket_qua.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("TONG HOP K-MEANS RANDOM VA K-MEANS++\n")
    f.write(tong_hop.round(3).to_string(index=False))
    f.write("\n\nChi tiet 100 lan chay: bai3_ket_qua_100_lan.csv\n")
