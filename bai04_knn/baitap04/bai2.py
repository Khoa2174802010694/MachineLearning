'''
Bài tập 2: Khoảng cách Manhattan
Chạy lại e3_chon_k.py nhưng thêm tham số metric="manhattan" vào
KNeighborsClassifier. Ghi lại k tốt nhất và accuracy trên tập kiểm tra, rồi so với
khoảng cách Euclid. Nhận xét một câu: đổi cách đo khoảng cách có làm kết quả khác đi
nhiều không?
'''
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Doc du lieu
df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

# Chia du lieu giong e3_chon_k.py
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tim k tot nhat bang kiem dinh cheo 5 phan
cac_k = range(1, 52, 2)
ket_qua = []

for metric in ["euclidean", "manhattan"]:
    diem = {}

    for k in cac_k:
        mo_hinh = KNeighborsClassifier(
            n_neighbors=k,
            metric=metric
        )
        diem[k] = cross_val_score(
            mo_hinh, X_tr, y_tr, cv=5
        ).mean()

    k_tot = max(diem, key=diem.get)

    # Danh gia tren tap kiem tra
    mo_hinh = KNeighborsClassifier(
        n_neighbors=k_tot,
        metric=metric
    )
    mo_hinh.fit(X_tr, y_tr)

    y_du_doan = mo_hinh.predict(X_te)
    accuracy = accuracy_score(y_te, y_du_doan)

    ket_qua.append({
        "Khoang cach": metric,
        "k tot nhat": k_tot,
        "CV trung binh": diem[k_tot],
        "Accuracy kiem tra": accuracy
    })

# In ket qua
print("SO SANH KHOANG CACH EUCLID VA MANHATTAN")
print(f"So mau hoc: {len(X_tr)}")
print(f"So mau kiem tra: {len(X_te)}")
print("Kiem dinh cheo: 5 phan, k tu 1 den 51 (buoc 2)")
print()

for dong in ket_qua:
    print(f"Khoang cach: {dong['Khoang cach']}")
    print(f" k tot nhat: {dong['k tot nhat']}")
    print(f" Diem CV: {dong['CV trung binh']:.4f}")
    print(f" Accuracy: {dong['Accuracy kiem tra']:.4f}")
    print()

# Nhan xet
chenh_lech = abs(
    ket_qua[0]["Accuracy kiem tra"]
    - ket_qua[1]["Accuracy kiem tra"]
)

print("NHAN XET")
print(f"Chenh lech accuracy: {chenh_lech:.4f}")

if chenh_lech < 0.02:
    print("Thay doi khoang cach khong lam ket qua khac biet nhieu.")
else:
    print("Thay doi khoang cach lam ket qua khac biet dang ke.")

# Luu ket qua
outputs = Path("outputs")
outputs.mkdir(parents=True, exist_ok=True)

pd.DataFrame(ket_qua).to_csv(
    outputs / "bai2_so_sanh_khoang_cach.csv",
    index=False,
    encoding="utf-8-sig"
)

noi_dung = (
    f"Euclidean: k={ket_qua[0]['k tot nhat']}, "
    f"Accuracy={ket_qua[0]['Accuracy kiem tra']:.4f}\n"
    f"Manhattan: k={ket_qua[1]['k tot nhat']}, "
    f"Accuracy={ket_qua[1]['Accuracy kiem tra']:.4f}\n"
    f"Chenh lech accuracy: {chenh_lech:.4f}\n"
    "Nhan xet: Thay doi cach do khoang cach co lam ket qua thay doi, "
    "nhung muc chenh lech khong lon."
)

(outputs / "bai2_nhan_xet.txt").write_text(
    noi_dung, encoding="utf-8"
)