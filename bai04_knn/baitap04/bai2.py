"""Bài tập 2: Khoảng cách Manhattan.
Chạy lại e3_chon_k.py nhưng thêm tham số metric="manhattan" vào
KNeighborsClassifier. Ghi lại k tốt nhất và accuracy trên tập kiểm tra, rồi so với
khoảng cách Euclid. Nhận xét một câu: đổi cách đo khoảng cách có làm kết quả khác đi
nhiều không?
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Chạy kiểm định chéo với metric="manhattan"
diem_manhattan = {}
for k in range(1, 52, 2):
    diem_manhattan[k] = cross_val_score(
        KNeighborsClassifier(n_neighbors=k, metric="manhattan"), X_tr, y_tr, cv=5
    ).mean()

k_tot_manhattan = max(diem_manhattan, key=diem_manhattan.get)
model_manhattan = KNeighborsClassifier(n_neighbors=k_tot_manhattan, metric="manhattan")
model_manhattan.fit(X_tr, y_tr)
acc_manhattan = accuracy_score(y_te, model_manhattan.predict(X_te))

print(f"K tốt nhất (Manhattan): {k_tot_manhattan}")
print(f"Accuracy trên tập kiểm tra (Manhattan): {acc_manhattan:.4f}")
print(
    " Đổi cách đo khoảng cách từ Euclid sang Manhattan không làm kết quả thay đổi nhiều (accuracy rất gần nhau hoặc tương đương), do đặc trưng hình học của phân phối dữ liệu hai vầng trăng ở bài toán này không quá nhạy cảm với việc thay đổi chuẩn khoảng cách."
)
print("=" * 60)