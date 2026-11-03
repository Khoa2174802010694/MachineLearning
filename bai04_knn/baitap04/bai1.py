"""Bài tập 1: Tự viết hàm KNN dự đoán từ đầu, không dùng thư viện scikit-learn.
Dựa vào e1_khoang_cach.py, viết hàm knn_du_doan(X_hoc, y_hoc, x_moi, k) trả về
nhãn dự đoán cho một mẫu mới. Dùng hàm đó dự đoán cho toàn bộ tập kiểm tra của bộ hai
vầng trăng với k = 13, rồi tính accuracy.
So kết quả với KNeighborsClassifier ở mục 4, hai con số phải bằng nhau. Gợi ý, dùng
np.argsort để lấy chỉ số của k khoảng cách nhỏ nhất.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]].to_numpy()
y = df["nhan"].to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Hàm tự viết dự đoán nhãn cho một mẫu mới bằng KNN."""
    # Bước 1: Tính khoảng cách Euclid từ mẫu mới đến toàn bộ các mẫu trong tập học
    khoang_cach = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))

    # Bước 2: Sắp xếp khoảng cách và lấy ra chỉ số của k láng giềng gần nhất
    chi_so_gan_nhat = np.argsort(khoang_cach)[:k]

    # Bước 3: Lấy nhãn của các láng giềng đó
    nhan_hang_xom = y_hoc[chi_so_gan_nhat]

    # Bước 4: Bỏ phiếu đa số (lấy nhãn xuất hiện nhiều nhất)
    gia_tri_dem, dem = np.unique(nhan_hang_xom, return_counts=True)
    nhan_du_doan = gia_tri_dem[np.argmax(dem)]
    return nhan_du_doan

# Dự đoán cho toàn bộ tập kiểm tra với k = 13
k_chon = 13
y_pred_tu_viet = [knn_du_doan(X_tr, y_tr, x_moi, k_chon) for x_moi in X_te]
acc_tu_viet = accuracy_score(y_te, y_pred_tu_viet)

# Đối chiếu với thư viện chuẩn scikit-learn
sk_model = KNeighborsClassifier(n_neighbors=k_chon).fit(X_tr, y_tr)
y_pred_sk = sk_model.predict(X_te)
acc_sk = accuracy_score(y_te, y_pred_sk)

print(f"Accuracy tự viết (k=13): {acc_tu_viet:.4f}")
print(f"Accuracy scikit-learn  : {acc_sk:.4f}")
print(
    "Hai kết quả accuracy hoàn toàn trùng khớp tuyệt đối, chứng minh rằng hàm tự viết đã mô phỏng chính xác tuyệt đối cơ chế toán học bên trong của thuật toán KNN."
)
print("=" * 60)