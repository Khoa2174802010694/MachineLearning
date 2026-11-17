'''
Bài tập 6: MAE và RMSE với một lần đoán hỏng
Trên dữ liệu giá nhà của bài 1, tính MAE, RMSE và R2
của hồi quy tuyến tính trên tập kiểm
tra. Sau đó cộng thêm 5 tỷ vào dự đoán của đúng một căn, rồi tính lại ba con số.
Con số nào thay đổi nhiều nhất? Giải thích vì sao RMSE phạt nặng sai số lớn hơn MAE.
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()
lr.fit(X_tr, y_tr)

du = lr.predict(X_te)

mae_1 = mean_absolute_error(y_te, du)
rmse_1 = np.sqrt(mean_squared_error(y_te, du))
r2_1 = r2_score(y_te, du)

print("Ket qua truoc khi them sai so:")
print(f"MAE: {mae_1:.4f}")
print(f"RMSE: {rmse_1:.4f}")
print(f"R2: {r2_1:.4f}")

du_moi = du.copy()
du_moi[0] += 5

mae_2 = mean_absolute_error(y_te, du_moi)
rmse_2 = np.sqrt(mean_squared_error(y_te, du_moi))
r2_2 = r2_score(y_te, du_moi)

print("\nKet qua sau khi cong them 5 ty vao mot du doan:")
print(f"MAE: {mae_2:.4f}")
print(f"RMSE: {rmse_2:.4f}")
print(f"R2: {r2_2:.4f}")
print("\nMuc thay doi:")
print(f"MAE: {mae_2 - mae_1:+.4f}")
print(f"RMSE: {rmse_2 - rmse_1:+.4f}")
print(f"R2: {r2_2 - r2_1:+.4f}")
