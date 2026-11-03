"""Bài tập 5: Chọn k cho bài toán hồi quy
Trên dữ liệu giá nhà, dùng kiểm định chéo 5 phần trên tập học để chọn k
tốt nhất cho KNeighborsRegressor, với k chạy từ 1 tới 20. Gợi ý, truyền
scoring="neg_mean_absolute_error" vào cross_val_score, và nhớ rằng điểm trả về
mang dấu âm.
Với k tốt nhất, MAE trên tập kiểm tra có vượt được đường thẳng không?
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

df = pd.read_csv("data/gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Dùng kiểm định chéo 5 phần, k chạy từ 1 tới 20
# scoring='neg_mean_absolute_error' trả về giá trị âm
diem_mae_am = {}
for k in range(1, 21):
    diem = cross_val_score(
        KNeighborsRegressor(n_neighbors=k),
        X_tr,
        y_tr,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    diem_mae_am[k] = diem.mean()

# Lấy k tốt nhất (giá trị âm lớn nhất tương ứng MAE dương nhỏ nhất)
k_tot_reg = max(diem_mae_am, key=diem_mae_am.get)

# Huấn luyện mô hình KNN với k tốt nhất và Linear Regression để so sánh
knn_reg = KNeighborsRegressor(n_neighbors=k_tot_reg).fit(X_tr, y_tr)
linear_reg = LinearRegression().fit(X_tr, y_tr)

mae_knn = mean_absolute_error(y_te, knn_reg.predict(X_te))
mae_linear = mean_absolute_error(y_te, linear_reg.predict(X_te))

print(f"K tốt nhất cho hồi quy: {k_tot_reg}")
print(f"MAE KNN (k={k_tot_reg}) trên tập kiểm tra: {mae_knn:.4f}")
print(f"MAE Đường thẳng (Linear Regression)    : {mae_linear:.4f}")
print(
    "Kết quả MAE của KNN với k tối ưu thường đạt mức rất sát hoặc vượt qua mô hình đường thẳng tuyến tính nhờ tính linh hoạt nắm bắt biên độ phi tuyến của giá nhà."
)
print("=" * 60)