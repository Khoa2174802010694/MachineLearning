# -*- coding: utf-8 -*-

"""
Bài tập 5: Tìm ngưỡng có F1 cao nhất.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
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


# Lấy xác suất qua môn
xac_suat = mo_hinh.predict_proba(X_test)[:, 1]


# Tạo các ngưỡng từ 0.05 đến 0.95
cac_nguong = np.arange(0.05, 1.0, 0.05)


print("Nguong    F1")
print("----------------")


# Lưu ngưỡng và F1 tốt nhất
nguong_tot_nhat = None
f1_tot_nhat = -1


# Thử từng ngưỡng
for nguong in cac_nguong:

    # Chuyển xác suất thành nhãn
    y_pred = (xac_suat >= nguong).astype(int)

    # Tính F1
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    print(f"{nguong:.2f}     {f1:.4f}")

    # Kiểm tra xem có phải F1 cao nhất hay không
    if f1 > f1_tot_nhat:
        f1_tot_nhat = f1
        nguong_tot_nhat = nguong


print()
print(f"Nguong co F1 cao nhat: {nguong_tot_nhat:.2f}")
print(f"F1 cao nhat: {f1_tot_nhat:.4f}")
print()


# Nhận xét
if nguong_tot_nhat == 0.5:
    print("Nhan xet: Nguong tot nhat theo F1 bang 0.5.")
else:
    print(
        "Nhan xet: Nguong tot nhat theo F1 khong bang 0.5."
    )