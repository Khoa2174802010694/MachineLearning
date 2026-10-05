# -*- coding: utf-8 -*-

"""
Bước 3: Để scikit-learn tìm w và b từ dữ liệu thật.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("data/sinh_vien.csv")


# X phải là bảng hai chiều nên sử dụng hai cặp ngoặc vuông.
# y là dãy một chiều nên chỉ sử dụng một cặp ngoặc vuông.
X = df[["gio_on"]]
y = df["qua_mon"]


# Khởi tạo và huấn luyện mô hình hồi quy logistic
mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)


# Lấy hệ số w và hệ số chặn b của mô hình
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])


print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()


# Số giờ ôn mà mô hình dự đoán xác suất qua môn bằng 0.5
gio_nguong = -b / w

print(f"So gio on ung voi xac suat 0.5: {gio_nguong:.2f} gio")
print()


# Tạo dữ liệu mới để kiểm tra mô hình
can_moi = pd.DataFrame({
    "gio_on": [5.0, 10.0, 13.0, 20.0, 28.0]
})


# Tính xác suất qua môn và nhãn dự đoán
xac_suat = mo_hinh.predict_proba(can_moi)[:, 1]
nhan = mo_hinh.predict(can_moi)


# Hiển thị kết quả dự đoán
print("Du doan cho nam ban moi:")
print(" So gio on    Xac suat qua    Nhan mo hinh dua ra")

for gio, p, n in zip(can_moi["gio_on"], xac_suat, nhan):
    print(f" {gio:7.1f}       {p:.4f}               {n}")