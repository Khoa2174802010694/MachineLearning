# -*- coding: utf-8 -*-

"""
Bài tập 3: Dự đoán kết quả cho một sinh viên cụ thể.

Mô hình chỉ sử dụng số giờ ôn để dự đoán khả năng qua môn.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression


# Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")


# Chọn biến đầu vào và biến mục tiêu
X = df[["gio_on"]]
y = df["qua_mon"]


# Huấn luyện mô hình hồi quy logistic
mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)


# Lấy hệ số của mô hình
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])


# Hàm dự đoán
def du_doan(gio):
    """
    Nhận số giờ ôn và in ra:
    - Giá trị z
    - Xác suất qua môn
    - Nhãn dự đoán theo ngưỡng 0.5
    """

    # Tính z = w*x + b
    z = w * gio + b

    # Tính xác suất bằng hàm sigmoid
    xac_suat = 1 / (1 + __import__("math").exp(-z))

    # Chuyển xác suất thành nhãn
    nhan = 1 if xac_suat >= 0.5 else 0

    print(f"So gio on: {gio:.2f}")
    print(f"z = {z:.4f}")
    print(f"Xac suat qua mon = {xac_suat:.4f}")
    print(f"Nhãn du doan = {nhan}")
    print()


# Dự đoán cho các số giờ được yêu cầu
for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)


# Giải thích trường hợp 12.89 giờ
print("Giai thich:")
print(
    "12.89 gio gan voi nguong phan chia cua mo hinh, "
    "nen gia tri z gan 0."
)
print(
    "Khi z gan 0 thi ham sigmoid cho xac suat gan 0.5, "
    "vi vay ket qua tai 12.89 gio gan muc 50%."
)