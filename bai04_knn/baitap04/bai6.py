"""Bài tập 6: Đo thời gian dự đoán
Tạo bộ dữ liệu ngẫu nhiên có 1000, 10000 và 100000 mẫu với 2 đặc trưng bằng
np.random.default_rng(0).random. Với mỗi cỡ, đo thời gian .fit và thời gian
.predict cho 1000 mẫu mới bằng mô đun time.
Lập bảng kết quả rồi giải thích vì sao KNN được gọi là mô hình học lười. Kết quả thời gian
khác nhau giữa các máy là chuyện bình thường, chỉ cần so tương đối giữa các cỡ dữ liệu.
"""

import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

co_du_lieu = [1000, 10000, 100000]

print(f"{'Cỡ mẫu (N)':<12} | {'Thời gian .fit (s)':<20} | {'Thời gian .predict (1000 mẫu)':<30}")
print("-" * 68)

rng = np.random.default_rng(0)

for n in co_du_lieu:
    # Tạo dữ liệu ngẫu nhiên có 2 đặc trưng
    X_train = rng.random((n, 2))
    y_train = rng.integers(0, 2, size=n)
    X_moi = rng.random((1000, 2))

    # Đo thời gian .fit
    bat_dau_fit = time.time()
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    thoi_gian_fit = time.time() - bat_dau_fit

    # Đo thời gian .predict cho 1000 mẫu mới
    bat_dau_pred = time.time()
    model.predict(X_moi)
    thoi_gian_pred = time.time() - bat_dau_pred

    print(f"{n:<12} | {thoi_gian_fit:<20.6f} | {thoi_gian_pred:<30.6f}")

print(
    "- Thời gian `.fit()` của KNN cực kỳ nhanh (gần như bằng 0) vì nó không hề tính toán trọng số hay tối ưu hóa tham số nào cả, mà chỉ đơn thuần lưu giữ toàn bộ dữ liệu vào RAM.\n"
    "- Ngược lại, thời gian `.predict()` tăng lên đáng kể theo quy mô dữ liệu N, vì với mỗi mẫu mới, mô hình phải quét và tính khoảng cách đến toàn bộ N điểm trong tập huấn luyện."
)