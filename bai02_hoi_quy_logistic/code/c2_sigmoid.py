# -*- coding: utf-8 -*-

"""
Bước 2: Làm quen với hàm sigmoid bằng tay trước khi gọi thư viện.
"""

import numpy as np


def sigmoid(z):
    """Ép một số thực bất kỳ về khoảng từ 0 đến 1."""
    return 1 / (1 + np.exp(-z))


# Hiển thị một số giá trị của hàm sigmoid
print("Bang gia tri cua ham sigmoid")
print(" z    sigmoid(z)")

for z in [-6, -4, -2, -1, 0, 1, 2, 4, 6]:
    print(f"{z:3d}    {sigmoid(z):.4f}")

print()


# Kiểm tra tính chất sigmoid(-z) = 1 - sigmoid(z)
print("Kiem tra sigmoid(-z) = 1 - sigmoid(z):")

for z in [1.0, 2.5, 3.7]:
    trai = sigmoid(-z)
    phai = 1 - sigmoid(z)

    print(f"z = {z}: {trai:.6f} va {phai:.6f}")

print()


# Thử với một mô hình giả định
# w = 0.4 và b = -5
w, b = 0.4, -5.0

print(f"Mo hinh gia dinh: w = {w}, b = {b}")
print(" So gio on    z = w*x + b    Xac suat qua mon")

for gio in [5, 10, 12.5, 15, 20, 25]:
    z = w * gio + b
    xac_suat = sigmoid(z)

    print(f" {gio:7.1f}    {z:10.2f}        {xac_suat:.4f}")