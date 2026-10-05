# -*- coding: utf-8 -*-

"""
Bước 1: Đọc bộ dữ liệu sinh viên và xem qua một lượt.

Chú thích trong tệp viết tiếng Việt có dấu, nhưng phần in ra màn hình
cố ý viết không dấu vì cửa sổ lệnh Windows có thể không hiển thị
tiếng Việt có dấu đúng cách.
"""

import pandas as pd


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("data/sinh_vien.csv")


# Hiển thị kích thước của bộ dữ liệu
print("Kich thuoc bang (so dong, so cot):", df.shape)
print()


# Hiển thị 5 dòng dữ liệu đầu tiên
print("Nam dong dau tien:")
print(df.head())
print()


# Đếm số sinh viên theo kết quả qua môn
# Cột "qua_mon" chỉ có hai giá trị: 0 và 1
print("So sinh vien theo ket qua:")
print(df["qua_mon"].value_counts())
print()


# Tính tỷ lệ sinh viên qua môn
print("Ty le qua mon:", round(df["qua_mon"].mean(), 4))
print()


# So sánh số giờ ôn trung bình giữa hai nhóm
print("So gio on trung binh theo nhom:")
print(df.groupby("qua_mon")["gio_on"].mean().round(2))