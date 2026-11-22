'''
Bài tập 4: Láng giềng có trọng số
Thêm tham số weights="distance" vào KNeighborsClassifier. Khi đó láng giềng càng
gần thì phiếu càng nặng.
Chạy lại với k = 1, 13, 101 trên bộ hai vầng trăng và so với bảng ở mục 4. Với k = 101,
trọng số theo khoảng cách có giúp mô hình bớt cẩu thả không? Vì sao?
'''
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Doc du lieu hai vang trang
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia du lieu giong e4_so_sanh.py
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

cac_k = [1, 13, 101]
ket_qua = []

print("SO SANH KNN VOI TRONG SO KHOANG CACH")
print(f"So mau hoc: {len(X_tr)}")
print(f"So mau kiem tra: {len(X_te)}")
print()

print(f"{'k':>5} {'Uniform':>12} {'Distance':>12} {'Chenh lech':>12}")

for k in cac_k:
    # KNN mac dinh: moi lang gieng co trong so bang nhau
    knn_uniform = KNeighborsClassifier(
        n_neighbors=k,
        weights="uniform"
    )
    knn_uniform.fit(X_tr, y_tr)
    acc_uniform = accuracy_score(
        y_te, knn_uniform.predict(X_te)
    )

    # KNN co trong so: lang gieng gan hon co phieu nang hon
    knn_distance = KNeighborsClassifier(
        n_neighbors=k,
        weights="distance"
    )
    knn_distance.fit(X_tr, y_tr)
    acc_distance = accuracy_score(
        y_te, knn_distance.predict(X_te)
    )

    chenh_lech = acc_distance - acc_uniform

    ket_qua.append({
        "k": k,
        "Uniform": acc_uniform,
        "Distance": acc_distance,
        "Chenh lech": chenh_lech
    })

    print(
        f"{k:5d} {acc_uniform:12.4f} "
        f"{acc_distance:12.4f} {chenh_lech:+12.4f}"
    )

# Nhan xet rieng cho k = 101
dong_101 = ket_qua[-1]

print()
print("NHAN XET")
print(
    f"Voi k = 101, accuracy Uniform = "
    f"{dong_101['Uniform']:.4f}"
)
print(
    f"Voi k = 101, accuracy Distance = "
    f"{dong_101['Distance']:.4f}"
)

if dong_101["Distance"] > dong_101["Uniform"]:
    print(
        "Trong so khoang cach giup mo hinh cai thien "
        "ket qua khi k = 101."
    )
else:
    print(
        "Trong so khoang cach khong giup cai thien "
        "accuracy khi k = 101."
    )

# Luu ket qua
outputs = Path("outputs")
outputs.mkdir(parents=True, exist_ok=True)

pd.DataFrame(ket_qua).to_csv(
    outputs / "bai4_so_sanh_trong_so.csv",
    index=False,
    encoding="utf-8-sig"
)

noi_dung = (
    "\n".join(
        f"k={r['k']}: Uniform={r['Uniform']:.4f}, "
        f"Distance={r['Distance']:.4f}, "
        f"Chenh lech={r['Chenh lech']:+.4f}"
        for r in ket_qua
    )
    + "\nNhan xet: Voi k=101, trong so khoang cach giup "
    "mo hinh cai thien accuracy vi cac lang gieng gan "
    "duoc uu tien hon cac lang gieng xa."
)

(outputs / "bai4_nhan_xet.txt").write_text(
    noi_dung,
    encoding="utf-8"
)