"""Bài tập 3: Đổi đơn vị đo
Trên bộ sinh_vien_mo_rong.csv, đổi cột chi_tieu từ nghìn đồng sang triệu đồng bằng
cách chia cho 1000. Chạy KNN với k = 7 không chuẩn hoá trên ba cột mới.
Accuracy thay đổi thế nào so với 0.5333 ở mục 3? Giải thích vì sao chỉ đổi đơn vị mà kết
quả lại khác hẳn. Nếu có chuẩn hoá thì đổi đơn vị có làm kết quả thay đổi không?
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/sinh_vien_mo_rong.csv")
y = df["qua_mon"]

# Đổi cột chi_tieu từ nghìn đồng sang triệu đồng (chia cho 1000)
df["chi_tieu_trieu"] = df["chi_tieu"] / 1000.0
ba_cot_moi = ["gio_on", "diem_giua_ky", "chi_tieu_trieu"]

X_tr, X_te, y_tr, y_te = train_test_split(
    df[ba_cot_moi], y, test_size=0.25, random_state=17, stratify=y
)

# Chạy KNN với k = 7 không chuẩn hóa trên ba cột mới
knn_khong_chuan = KNeighborsClassifier(n_neighbors=7)
knn_khong_chuan.fit(X_tr, y_tr)
acc_moi = accuracy_score(y_te, knn_khong_chuan.predict(X_te))

print(f"Accuracy sau khi đổi đơn vị (không chuẩn hóa): {acc_moi:.4f}")
print(
    "- Việc đổi đơn vị làm thay đổi hoàn toàn độ lớn tỷ lệ giữa các đặc trưng, khiến mô hình bắt, gán khoảng cách lệch lạc đi."
    "\n"
    "- Nếu hệ thống có áp dụng bước chuẩn hóa (StandardScaler), kết quả accuracy sẽ không thay đổi vì phép chuẩn hóa đã tự động quy đổi mọi biến về cùng một miền giá trị thuần nhất."
)
print("=" * 60)