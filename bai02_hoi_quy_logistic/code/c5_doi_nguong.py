# -*- coding: utf-8 -*-

"""
Bước 5: Tự áp ngưỡng và xem precision với recall thay đổi ra sao.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split


# Đọc dữ liệu từ tệp CSV
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


# Huấn luyện mô hình hồi quy logistic
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# predict_proba trả về hai cột:
# cột 0 cho lớp 0 và cột 1 cho lớp 1.
p = mo_hinh.predict_proba(X_test)[:, 1]


# Thử nhiều ngưỡng khác nhau
print("Nguong   So ban bi doan la qua   Precision   Recall")

for nguong in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # Tự chuyển xác suất thành nhãn 0 hoặc 1
    y_pred = (p >= nguong).astype(int)

    # Tính Precision và Recall
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=1,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(
        f" {nguong:.1f}"
        f"{y_pred.sum():18d}"
        f"{precision:13.4f}"
        f"{recall:9.4f}"
    )

print()


# Nhận xét xu hướng khi thay đổi ngưỡng
print("Doc bang tren tu duoi len:")
print("Nguong cang cao thi mo hinh cang kho du doan la qua,")
print("precision thuong tang nhung recall co xu huong giam.")
print("Nguong thap thi mo hinh de du doan la qua hon,")
print("recall thuong tang nhung precision co xu huong giam.")