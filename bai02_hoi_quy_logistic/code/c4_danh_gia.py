# -*- coding: utf-8 -*-

"""
Bước 4: Chia dữ liệu rồi chấm điểm mô hình bằng bốn thước đo.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
# stratify=y giúp giữ đúng tỷ lệ qua và rớt ở cả hai tập.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)


print("So sinh vien de hoc:", len(X_train))
print("So sinh vien de kiem tra:", len(X_test))
print()


# Khởi tạo và huấn luyện mô hình
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Dự đoán kết quả trên tập kiểm tra
y_pred = mo_hinh.predict(X_test)


# Lấy bốn giá trị từ ma trận nhầm lẫn.
# Thứ tự của scikit-learn là: TN, FP, FN, TP.
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Ma tran nham lan:")
print(f" TN = {tn:2d} | Doan rot, that su rot")
print(f" FP = {fp:2d} | Doan qua, that ra rot")
print(f" FN = {fn:2d} | Doan rot, that ra qua")
print(f" TP = {tp:2d} | Doan qua, that su qua")
print()


# Tính bốn chỉ số đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")
print()


# Tự tính lại bằng công thức để đối chiếu với thư viện
print("Tu tinh lai bang cong thuc:")

n = len(y_test)

print(
    f"Accuracy  = ({tp} + {tn}) / {n} "
    f"= {(tp + tn) / n:.4f}"
)

print(
    f"Precision = {tp} / ({tp} + {fp}) "
    f"= {tp / (tp + fp):.4f}"
)

print(
    f"Recall    = {tp} / ({tp} + {fn}) "
    f"= {tp / (tp + fn):.4f}"
)