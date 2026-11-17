'''
Bài tập 2: Một mạng XOR khác
Kiểm chứng bằng numpy rằng mạng sau cũng tính đúng XOR:
h1 = max(0, x1 - x2) h2 = max(0, x2 - x1) y = h1 + h2
Giải thích bằng lời vì sao mạng này đúng
'''
import os
import numpy as np
import pandas as pd

os.makedirs("outputs", exist_ok=True)

def relu(z):
    return np.maximum(0, z)

# Bốn trường hợp đầu vào của phép XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Tính hai nơ-ron ẩn
x1 = X[:, 0]
x2 = X[:, 1]

h1 = relu(x1 - x2)
h2 = relu(x2 - x1)

# Tính đầu ra
y = h1 + h2

# Kết quả XOR mong đợi
y_xor = np.array([0, 1, 1, 0])

print("KIEM CHUNG MANG XOR")
print("x1 x2 | h1 h2 | y | XOR")

for i in range(len(X)):
    print(
        f"{X[i, 0]:.0f}  "
        f"{X[i, 1]:.0f}  | "
        f"{h1[i]:.0f}  "
        f"{h2[i]:.0f}  | "
        f"{y[i]:.0f} | "
        f"{y_xor[i]}"
    )

# Kiểm tra kết quả
if np.array_equal(y.astype(int), y_xor):
    print("\nKet luan: Mang tinh dung phep XOR.")
else:
    print("\nKet luan: Mang chua tinh dung phep XOR.")

# Lưu kết quả
ket_qua = pd.DataFrame({
    "x1": X[:, 0],
    "x2": X[:, 1],
    "h1": h1,
    "h2": h2,
    "y": y,
    "xor_mong_doi": y_xor
})

ket_qua.to_csv(
    "outputs/bai2_ket_qua.csv",
    index=False
)

