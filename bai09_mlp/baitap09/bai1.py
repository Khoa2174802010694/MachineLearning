'''
Bài tập 1: Tự dự đoán bằng trọng số đã học
Học lại mạng 8 nơ-ron ẩn trên bộ XOR như mục 3. Lấy mlp.coefs_ và mlp.intercepts_,
rồi tự tính lan truyền tiến bằng numpy cho 5 điểm đầu của tập kiểm tra: lớp ẩn dùng ReLU,
lớp ra dùng sigmoid.
So kết quả với mlp.predict_proba, cột thứ hai. Hai kết quả phải khớp tới nhiều chữ số
thập phân.
'''
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
os.makedirs("outputs", exist_ok=True)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

# Đọc dữ liệu XOR và chia tập huấn luyện, kiểm tra
df = pd.read_csv("data/xor.csv")

X = df[["x1", "x2"]].to_numpy(dtype=float)
y = df["nhan"].to_numpy()

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Huấn luyện mạng MLP có 8 nơ-ron ẩn
mlp = MLPClassifier(
    hidden_layer_sizes=(8,),
    solver="lbfgs",
    max_iter=5000,
    random_state=0
)

mlp.fit(X_tr, y_tr)

# Lấy trọng số và bias đã học
W1 = mlp.coefs_[0]
b1 = mlp.intercepts_[0]

W2 = mlp.coefs_[1]
b2 = mlp.intercepts_[1]

# Tự tính lan truyền tiến bằng NumPy cho 5 điểm đầu
X_5 = X_te[:5]

H = relu(X_5 @ W1 + b1)
Z = H @ W2 + b2
Y_numpy = sigmoid(Z).ravel()

# Lấy xác suất dự đoán từ sklearn, cột thứ hai là xác suất lớp 1
Y_sklearn = mlp.predict_proba(X_5)[:, 1]

# In kết quả so sánh
print("SO SANH KET QUA NUMPY VA SKLEARN")
print("X1 X2 | NumPy | sklearn | Sai so")

for i in range(len(X_5)):
    sai_so = abs(Y_numpy[i] - Y_sklearn[i])

    print(
        f"{X_5[i, 0]:.0f}  "
        f"{X_5[i, 1]:.0f}  | "
        f"{Y_numpy[i]:.10f} | "
        f"{Y_sklearn[i]:.10f} | "
        f"{sai_so:.12f}"
    )

# Kiểm tra sai số lớn nhất giữa hai cách tính
sai_so_lon_nhat = np.max(np.abs(Y_numpy - Y_sklearn))

print()
print(f"Sai so lon nhat: {sai_so_lon_nhat:.12f}")

if np.allclose(Y_numpy, Y_sklearn, atol=1e-7):
    print("Ket luan: Hai ket qua khop nhau.")
else:
    print("Ket luan: Hai ket qua chua khop nhau.")

# Lưu kết quả ra tệp
ket_qua = pd.DataFrame({
    "x1": X_5[:, 0],
    "x2": X_5[:, 1],
    "xac_suat_numpy": Y_numpy,
    "xac_suat_sklearn": Y_sklearn,
    "sai_so": np.abs(Y_numpy - Y_sklearn)
})

ket_qua.to_csv(
    "outputs/bai1_so_sanh.csv",
    index=False
)
