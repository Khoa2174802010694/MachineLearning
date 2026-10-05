# -*- coding: utf-8 -*-

"""
Bước 6: Thêm điểm giữa kỳ làm biến thứ hai.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("data/sinh_vien.csv")

y = df["qua_mon"]


# So sánh mô hình một biến và hai biến
# Chỉ thay đổi danh sách cột được đưa vào X.
for ten, cot in [
    ("Mot bien", ["gio_on"]),
    ("Hai bien", ["gio_on", "diem_giua_ky"]),
]:
    X = df[cot]

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

    # Tính độ chính xác trên tập kiểm tra
    do_chinh_xac = accuracy_score(
        y_test,
        mo_hinh.predict(X_test),
    )

    print(
        f"{ten}: accuracy tren tap kiem tra = "
        f"{do_chinh_xac:.4f}"
    )

print()


# Huấn luyện lại mô hình hai biến để xem kỹ các hệ số
X = df[["gio_on", "diem_giua_ky"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)

mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Hiển thị hệ số của từng biến
print("He so cua tung bien:")

for ten_cot, he_so in zip(X.columns, mo_hinh.coef_[0]):
    print(f" {ten_cot:14s} {he_so:+.4f}")

print(f" {'he so chan':14s} {mo_hinh.intercept_[0]:+.4f}")
print()


# Nhận xét về dấu của các hệ số
print("Ca hai he so deu duong, nghia la on nhieu hon va")
print("diem giua ky cao hon deu lam tang xac suat qua mon.")