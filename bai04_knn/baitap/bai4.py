"""Bài tập 4: Láng giềng có trọng số
Thêm tham số weights="distance" vào KNeighborsClassifier. Khi đó láng giềng càng
gần thì phiếu càng nặng.
Chạy lại với k = 1, 13, 101 trên bộ hai vầng trăng và so với bảng ở mục 4. Với k = 101,
trọng số theo khoảng cách có giúp mô hình bớt cẩu thả không? Vì sao?
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

danh_sach_k = [1, 13, 101]

print("So sánh Accuracy giữa Uniform và Weighted (weights='distance'):")
print(f"{'k':<5} | {'Uniform':<10} | {'Distance Weighted':<18}")
print("-" * 40)

for k in danh_sach_k:
    # Không dùng trọng số (mặc định uniform)
    m_uni = KNeighborsClassifier(n_neighbors=k, weights="uniform").fit(X_tr, y_tr)
    acc_uni = accuracy_score(y_te, m_uni.predict(X_te))

    # Có dùng trọng số theo khoảng cách
    m_dist = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_tr, y_tr)
    acc_dist = accuracy_score(y_te, m_dist.predict(X_te))

    print(f"{k:<5} | {acc_uni:<10.4f} | {acc_dist:<18.4f}")

print(
    "\nVới k = 101 rất lớn, việc dùng trọng số theo khoảng cách (weights='distance') giúp các láng giềng ở gần có quyền lực lớn hơn hẳn các láng giềng ở xa, qua đó ngăn chặn tình trạng mô hình bị 'cẩu thả' (bị làm loãng nhãn bởi quá nhiều điểm hàng xóm ở tận đẩu tận đâu)."
)
print("=" * 60)