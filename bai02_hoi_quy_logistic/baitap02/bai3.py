'''
Bài tập 3: Dự đoán cho một bạn cụ thể
Khớp lại mô hình một biến như ở mục 4, rồi viết một hàm du_doan(gio) nhận vào số giờ
ôn và in ra ba thứ. Giá trị z, xác suất qua môn, và nhãn theo ngưỡng 0.5.
Gọi hàm đó với các giá trị 3, 8, 12.89, 18 và 26 giờ. Giải thích vì sao với 12.89 giờ thì xác
suất ra gần đúng 0.5.
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

# Chuan bi du lieu huan luyen
X = df[["gio_on"]]
y = df["qua_mon"]

# Huan luyen mo hinh hoi quy logistic
mo_hinh = LogisticRegression()
mo_hinh.fit(X, y)

# Lay he so w va b
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])

print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")


# Ham du doan ket qua qua mon
def du_doan(gio):
    z = w * gio + b
    xac_suat = 1 / (1 + np.exp(-z))
    nhan = 1 if xac_suat >= 0.5 else 0

    print(f"So gio on: {gio} gio")
    print(f"Gia tri z: {z:.6f}")
    print(f"Xac suat qua mon: {xac_suat:.4f}")
    print(f"Nhan du doan: {nhan}")
    print("-" * 40)


# Du doan cho nam gia tri gio on
print("\nKET QUA DU DOAN")
print("=" * 40)

for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)


# Giai thich moc 12.89 gio
print("GIAI THICH")
print(
    "Tai 12.89 gio, gia tri z gan bang 0 nen xac suat "
    "qua mon gan bang 0.5."
)
print(
    "Day la diem nguong phan loai cua mo hinh, "
    "vi sigmoid(0) = 0.5."
)
