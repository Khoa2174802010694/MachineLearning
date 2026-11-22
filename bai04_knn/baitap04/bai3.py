'''
Bài tập 3: Đổi đơn vị đo
Trên bộ sinh_vien_mo_rong.csv, đổi cột chi_tieu từ nghìn đồng sang triệu đồng bằng
cách chia cho 1000. Chạy KNN với k = 7 không chuẩn hoá trên ba cột mới.
Accuracy thay đổi thế nào so với 0.5333 ở mục 3? Giải thích vì sao chỉ đổi đơn vị mà kết
quả lại khác hẳn. Nếu có chuẩn hoá thì đổi đơn vị có làm kết quả thay đổi không?
'''
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Doc du lieu
df = pd.read_csv("data/sinh_vien_mo_rong.csv")

y = df["qua_mon"]
cac_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]

# Du lieu ban dau: chi_tieu tinh bang nghin dong
X_ban_dau = df[cac_cot].copy()

# Doi chi_tieu tu nghin dong sang trieu dong
X_doi_don_vi = X_ban_dau.copy()
X_doi_don_vi["chi_tieu"] = X_doi_don_vi["chi_tieu"] / 1000


def chia_du_lieu(X):
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=17,
        stratify=y
    )


def cham_diem(mo_hinh, X):
    X_tr, X_te, y_tr, y_te = chia_du_lieu(X)
    mo_hinh.fit(X_tr, y_tr)
    y_du_doan = mo_hinh.predict(X_te)
    return accuracy_score(y_te, y_du_doan)


# 1. KNN tren du lieu ban dau, khong chuan hoa
acc_ban_dau = cham_diem(
    KNeighborsClassifier(n_neighbors=7),
    X_ban_dau
)

# 2. KNN sau khi doi don vi, khong chuan hoa
acc_doi_don_vi = cham_diem(
    KNeighborsClassifier(n_neighbors=7),
    X_doi_don_vi
)

# 3. KNN co chuan hoa tren du lieu ban dau
acc_chuan_hoa = cham_diem(
    make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7)
    ),
    X_ban_dau
)

# 4. KNN co chuan hoa sau khi doi don vi
acc_doi_va_chuan_hoa = cham_diem(
    make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=7)
    ),
    X_doi_don_vi
)

# In ket qua
print("KNN VOI CAC THANG DO KHAC NHAU")
print(" k = 7")
print()
print(f" Ban dau, khong chuan hoa: {acc_ban_dau:.4f}")
print(f" Doi don vi, khong chuan hoa: {acc_doi_don_vi:.4f}")
print(f" Ban dau, co chuan hoa: {acc_chuan_hoa:.4f}")
print(f" Doi don vi, co chuan hoa: {acc_doi_va_chuan_hoa:.4f}")
print()

print("MUC THAY DOI")
print(
    f" Accuracy tang: "
    f"{acc_doi_don_vi - acc_ban_dau:+.4f}"
)

print()
print("NHAN XET")
print(
    "Khi khong chuan hoa, doi don vi lam thay doi "
    "muc do anh huong cua tung dac trung den khoang cach."
)
print(
    "Khi chuan hoa, doi don vi khong lam thay doi "
    "dang ke ket qua KNN."
)

# Luu ket qua
outputs = Path("outputs")
outputs.mkdir(parents=True, exist_ok=True)

bang_ket_qua = pd.DataFrame({
    "Truong hop": [
        "Ban dau, khong chuan hoa",
        "Doi don vi, khong chuan hoa",
        "Ban dau, co chuan hoa",
        "Doi don vi, co chuan hoa"
    ],
    "Accuracy": [
        acc_ban_dau,
        acc_doi_don_vi,
        acc_chuan_hoa,
        acc_doi_va_chuan_hoa
    ]
})

bang_ket_qua.to_csv(
    outputs / "bai3_so_sanh_don_vi.csv",
    index=False,
    encoding="utf-8-sig"
)

noi_dung = (
    f"Ban dau, khong chuan hoa: {acc_ban_dau:.4f}\n"
    f"Doi don vi, khong chuan hoa: {acc_doi_don_vi:.4f}\n"
    f"Ban dau, co chuan hoa: {acc_chuan_hoa:.4f}\n"
    f"Doi don vi, co chuan hoa: {acc_doi_va_chuan_hoa:.4f}\n"
    "Nhan xet: KNN rat nhay voi thang do khi khong chuan hoa. "
    "StandardScaler dua cac dac trung ve cung thang do, "
    "nen viec doi don vi truoc do khong lam thay doi ket qua."
)

(outputs / "bai3_nhan_xet.txt").write_text(
    noi_dung, encoding="utf-8"
)