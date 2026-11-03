# -*- coding: utf-8 -*-

"""
Bài tập 6: Đổi lớp dương thành lớp rớt môn.

So sánh Precision và Recall khi:
- Lớp dương là 1: qua môn.
- Lớp dương là 0: rớt môn.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


# Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)


# Huấn luyện mô hình
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Dự đoán kết quả
y_pred = mo_hinh.predict(X_test)


# Precision và Recall khi lớp dương là 1: qua môn
precision_lop_1 = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)

recall_lop_1 = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0,
)


# Precision và Recall khi lớp dương là 0: rớt môn
precision_lop_0 = precision_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0,
)

recall_lop_0 = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0,
)


# In kết quả
print("Lop 1 = qua mon")
print(f"Precision = {precision_lop_1:.4f}")
print(f"Recall    = {recall_lop_1:.4f}")
print()


print("Lop 0 = rot mon")
print(f"Precision = {precision_lop_0:.4f}")
print(f"Recall    = {recall_lop_0:.4f}")
print()


# Giải thích
print("Giai thich:")
print(
    "Khi doi lop duong tu 1 sang 0, cach tinh Precision "
    "va Recall cung thay doi theo lop dang duoc quan tam."
)
print(
    "Vi vay hai bo so co the khac nhau, du mo hinh va du lieu "
    "khong thay doi."
)
print(
    "Lop duong la lop ma ta dang xem la ket qua can tim, "
    "nen Precision va Recall phai duoc hieu theo lop do."
)