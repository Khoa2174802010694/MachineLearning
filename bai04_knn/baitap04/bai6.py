'''
Bài tập 6: Đo thời gian dự đoán
Tạo bộ dữ liệu ngẫu nhiên có 1000, 10000 và 100000 mẫu với 2 đặc trưng bằng
np.random.default_rng(0).random. Với mỗi cỡ, đo thời gian .fit và thời gian
.predict cho 1000 mẫu mới bằng mô đun time.
Lập bảng kết quả rồi giải thích vì sao KNN được gọi là mô hình học lười. Kết quả thời gian
khác nhau giữa các máy là chuyện bình thường, chỉ cần so tương đối giữa các cỡ dữ liệu.
'''
import os
import time
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

# Khoi tao bo sinh so ngau nhien
rng = np.random.default_rng(0)

# Cac kich thuoc du lieu can thu nghiem
cac_kich_thuoc = [1000, 10000, 100000]

# So mau moi dung de du doan
SO_MAU_MOI = 1000

# So lang gieng gan nhat
K = 5

ket_qua = []

for n in cac_kich_thuoc:

    # Tao du lieu huan luyen gom 2 dac trung
    X = rng.random((n, 2))

    # Tao nhan phan loai tu hai dac trung
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # Tao 1000 mau moi
    X_moi = rng.random((SO_MAU_MOI, 2))

    # Khoi tao mo hinh KNN
    mo_hinh = KNeighborsClassifier(n_neighbors=K)

    # Do thoi gian fit
    bat_dau = time.perf_counter()
    mo_hinh.fit(X, y)
    thoi_gian_fit = time.perf_counter() - bat_dau

    # Do thoi gian predict
    bat_dau = time.perf_counter()
    du_doan = mo_hinh.predict(X_moi)
    thoi_gian_predict = time.perf_counter() - bat_dau

    ket_qua.append({
        "So_mau": n,
        "So_dac_trung": 2,
        "So_mau_du_doan": SO_MAU_MOI,
        "Thoi_gian_fit_s": thoi_gian_fit,
        "Thoi_gian_predict_s": thoi_gian_predict
    })

# Tong hop ket qua
bang_ket_qua = pd.DataFrame(ket_qua)

print("KET QUA DO THOI GIAN KNN")
print("-" * 75)
print(bang_ket_qua.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# Luu ket qua
os.makedirs("outputs", exist_ok=True)

bang_ket_qua.to_csv(
    "outputs/bai6_thoi_gian_knn.csv",
    index=False,
    encoding="utf-8-sig"
)