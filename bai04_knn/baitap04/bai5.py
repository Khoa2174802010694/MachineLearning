'''
Bài tập 5: Chọn k cho bài toán hồi quy
Trên dữ liệu giá nhà, dùng kiểm định chéo 5 phần trên tập học để chọn k
tốt nhất cho KNeighborsRegressor, với k chạy từ 1 tới 20. Gợi ý, truyền
scoring="neg_mean_absolute_error" vào cross_val_score, và nhớ rằng điểm trả về
mang dấu âm.
Với k tốt nhất, MAE trên tập kiểm tra có vượt được đường thẳng không?
'''
import os
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Doc du lieu
df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich"]]
y = df["gia"]

# Chia tap hoc va tap kiem tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Tim k tot nhat bang kiem dinh cheo 5 phan
ket_qua = []

for k in range(1, 21):
    mo_hinh = KNeighborsRegressor(n_neighbors=k)

    scores = cross_val_score(
        mo_hinh,
        X_tr,
        y_tr,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    # Chuyen diem am thanh MAE duong
    mae_cv = -scores.mean()

    ket_qua.append({
        "k": k,
        "MAE_CV": mae_cv
    })

# Chon k co MAE CV nho nhat
bang_k = pd.DataFrame(ket_qua)
k_tot_nhat = int(bang_k.loc[bang_k["MAE_CV"].idxmin(), "k"])

# Huan luyen KNN voi k tot nhat
knn = KNeighborsRegressor(n_neighbors=k_tot_nhat)
knn.fit(X_tr, y_tr)

# Huan luyen hoi quy tuyen tinh de so sanh
duong_thang = LinearRegression()
duong_thang.fit(X_tr, y_tr)

# Danh gia tren tap kiem tra
du_doan_knn = knn.predict(X_te)
du_doan_duong = duong_thang.predict(X_te)

mae_knn = mean_absolute_error(y_te, du_doan_knn)
mae_duong = mean_absolute_error(y_te, du_doan_duong)

# In ket qua
print("KET QUA CHON K CHO KNN HOI QUY")
print("-" * 45)

print("\nMAE kiem dinh cheo 5 phan:")
print(bang_k.to_string(index=False))

print("\nK tot nhat:", k_tot_nhat)
print(f"MAE KNN tren tap kiem tra: {mae_knn:.4f}")
print(f"MAE hoi quy tuyen tinh:    {mae_duong:.4f}")

print("\nNhan xet:")
if mae_knn < mae_duong:
    print("KNN co MAE thap hon hoi quy tuyen tinh.")
else:
    print("KNN co MAE cao hon hoi quy tuyen tinh.")

# Luu ket qua
os.makedirs("outputs", exist_ok=True)

bang_k.to_csv(
    "outputs/bai5_chon_k.csv",
    index=False,
    encoding="utf-8-sig"
)

bang_so_sanh = pd.DataFrame({
    "Mo_hinh": ["KNN", "Hoi quy tuyen tinh"],
    "k": [k_tot_nhat, None],
    "MAE_test": [mae_knn, mae_duong]
})

bang_so_sanh.to_csv(
    "outputs/bai5_so_sanh_mo_hinh.csv",
    index=False,
    encoding="utf-8-sig"
)