'''
Bài tập 1: Tự viết KNN không dùng thư viện
Dựa vào e1_khoang_cach.py, viết hàm knn_du_doan(X_hoc, y_hoc, x_moi, k) trả về
nhãn dự đoán cho một mẫu mới. Dùng hàm đó dự đoán cho toàn bộ tập kiểm tra của bộ hai
vầng trăng với k = 13, rồi tính accuracy.
So kết quả với KNeighborsClassifier ở mục 4, hai con số phải bằng nhau. Gợi ý, dùng
np.argsort để lấy chỉ số của k khoảng cách nhỏ nhất.
'''
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Du doan nhan bang bo phieu cua k lang gieng gan nhat."""
    X_hoc = np.asarray(X_hoc, dtype=float)
    y_hoc = np.asarray(y_hoc)

    # Tinh khoang cach Euclid tu mau moi den toan bo mau hoc
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))

    # Lay chi so cua k mau co khoang cach nho nhat
    chi_so_gan_nhat = np.argsort(khoang_cach)[:k]

    # Bo phieu da so; neu hoa, chon nhan nho hon nhu sklearn
    cac_nhan, so_phieu = np.unique(
        y_hoc[chi_so_gan_nhat], return_counts=True
    )
    return cac_nhan[np.argmax(so_phieu)]


def main():
    # Doc du lieu tu thu muc goc cua bai thuc hanh
    df = pd.read_csv("data/hai_vang_trang.csv")
    X = df[["x1", "x2"]].to_numpy(dtype=float)
    y = df["nhan"].to_numpy()

    # Giu dung cach chia du lieu trong e3_chon_k.py va e4_so_sanh.py
    X_hoc, X_kiem_tra, y_hoc, y_kiem_tra = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=17,
        stratify=y
    )

    k = 13

    # Du doan bang ham KNN tu viet
    y_du_doan_tu_viet = np.array([
        knn_du_doan(X_hoc, y_hoc, x_moi, k)
        for x_moi in X_kiem_tra
    ])

    # Mo hinh sklearn de doi chieu voi cung tap hoc va tap kiem tra
    mo_hinh = KNeighborsClassifier(n_neighbors=k)
    mo_hinh.fit(X_hoc, y_hoc)
    y_du_doan_sklearn = mo_hinh.predict(X_kiem_tra)

    acc_tu_viet = accuracy_score(y_kiem_tra, y_du_doan_tu_viet)
    acc_sklearn = accuracy_score(y_kiem_tra, y_du_doan_sklearn)
    so_du_doan_khac_nhau = int(
        np.sum(y_du_doan_tu_viet != y_du_doan_sklearn)
    )

    print("KNN TU VIET VA KNeighborsClassifier")
    print(f" So mau hoc: {len(X_hoc)}")
    print(f" So mau kiem tra: {len(X_kiem_tra)}")
    print(f" k = {k}")
    print(f" Accuracy KNN tu viet: {acc_tu_viet:.4f}")
    print(f" Accuracy sklearn: {acc_sklearn:.4f}")
    print(f" So du doan khac nhau: {so_du_doan_khac_nhau}")
    print(
        " Ket qua: "
        + ("TRUNG KHOP" if np.isclose(acc_tu_viet, acc_sklearn)
           and so_du_doan_khac_nhau == 0 else "KHONG TRUNG KHOP")
    )

    # Luu bang doi chieu de phuc vu bao cao
    outputs = Path("outputs")
    outputs.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "nhan_thuc_te": y_kiem_tra,
        "du_doan_knn_tu_viet": y_du_doan_tu_viet,
        "du_doan_sklearn": y_du_doan_sklearn,
        "trung_khop": y_du_doan_tu_viet == y_du_doan_sklearn
    }).to_csv(outputs / "bai1_du_doan.csv", index=False, encoding="utf-8-sig")

    noi_dung = (
        f"So mau hoc: {len(X_hoc)}\n"
        f"So mau kiem tra: {len(X_kiem_tra)}\n"
        f"k: {k}\n"
        f"Accuracy KNN tu viet: {acc_tu_viet:.4f}\n"
        f"Accuracy sklearn: {acc_sklearn:.4f}\n"
        f"So du doan khac nhau: {so_du_doan_khac_nhau}\n"
        "Nhan xet: Hai cach cai dat cho ket qua trung khop."
        if so_du_doan_khac_nhau == 0
        else "Nhan xet: Hai cach cai dat co du doan khac nhau, can kiem tra lai."
    )
    (outputs / "bai1_nhan_xet.txt").write_text(noi_dung, encoding="utf-8")


if __name__ == "__main__":
    main()
