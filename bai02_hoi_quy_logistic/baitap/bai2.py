# -*- coding: utf-8 -*-

"""
Bài tập 2: Vẽ hàm sigmoid.

Vẽ hàm sigmoid trong khoảng z từ -8 đến 8,
đồng thời đánh dấu mức xác suất 0.5 và z = 0.
"""

import numpy as np
import matplotlib.pyplot as plt


# Tạo các giá trị z từ -8 đến 8
z = np.linspace(-8, 8, 400)


# Tính giá trị sigmoid
sigmoid = 1 / (1 + np.exp(-z))


# Vẽ đồ thị hàm sigmoid
plt.plot(z, sigmoid, label="Sigmoid")


# Vẽ đường ngang tại mức 0.5
plt.axhline(
    y=0.5,
    linestyle="--",
    label="y = 0.5",
)


# Vẽ đường dọc tại z = 0
plt.axvline(
    x=0,
    linestyle="--",
    label="z = 0",
)


# Đặt tiêu đề và tên các trục
plt.title("Ham Sigmoid")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")


# Hiển thị chú thích
plt.legend()


# Hiển thị lưới
plt.grid(True)


# Lưu hình ảnh
plt.savefig("baitap02/sigmoid.png", dpi=300)


# Hiển thị đồ thị
plt.show()