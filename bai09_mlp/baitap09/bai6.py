'''
Bài tập 6: Mạng nơ-ron cho hồi quy
Trên dữ liệu giá nhà của bài 1, dùng MLPRegressor(hidden_layer_sizes=(20,),
max_iter=5000, random_state=0) có chuẩn hoá. So MAE trên tập kiểm tra với hồi quy
tuyến tính. Dự đoán giá căn 250 m2
. Mạng nơ-ron có ngoại suy tốt như đường thẳng không?
'''
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

# Đọc dữ liệu giá nhà
df = pd.read_csv("data/gia_nha.csv")

X = df[["dien_tich", "so_phong", "tuoi_nha"]]
y = df["gia"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17
)

# Huấn luyện hồi quy tuyến tính
hoi_quy = LinearRegression()
hoi_quy.fit(X_tr, y_tr)

# Huấn luyện mạng nơ-ron có chuẩn hóa
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(20,),
        max_iter=5000,
        random_state=0
    )
)

mlp.fit(X_tr, y_tr)

# Đánh giá MAE trên tập kiểm tra
du_doan_tuyen_tinh = hoi_quy.predict(X_te)
du_doan_mlp = mlp.predict(X_te)

mae_tuyen_tinh = mean_absolute_error(
    y_te,
    du_doan_tuyen_tinh
)

mae_mlp = mean_absolute_error(
    y_te,
    du_doan_mlp
)

print("SO SANH HAI MO HINH HOI QUY")
print(f"MAE hoi quy tuyen tinh: {mae_tuyen_tinh:.4f}")
print(f"MAE MLP: {mae_mlp:.4f}")

# Tạo căn nhà có diện tích 250 m2
nha_moi = pd.DataFrame([{
    "dien_tich": 250,
    "so_phong": X["so_phong"].median(),
    "tuoi_nha": X["tuoi_nha"].median()
}])

gia_tuyen_tinh = hoi_quy.predict(nha_moi)[0]
gia_mlp = mlp.predict(nha_moi)[0]

print()
print("DU DOAN GIA CAN NHA 250 M2")
print("So phong su dung:", nha_moi["so_phong"].iloc[0])
print("Tuoi nha su dung:", nha_moi["tuoi_nha"].iloc[0])
print(f"Gia du doan hoi quy tuyen tinh: {gia_tuyen_tinh:.4f}")
print(f"Gia du doan MLP: {gia_mlp:.4f}")

# Kiểm tra diện tích 250 m2 có nằm ngoài phạm vi dữ liệu không
dien_tich_min = X["dien_tich"].min()
dien_tich_max = X["dien_tich"].max()

print()
print("PHAM VI DIEN TICH TRONG DU LIEU")
print(f"Dien tich nho nhat: {dien_tich_min:.2f} m2")
print(f"Dien tich lon nhat: {dien_tich_max:.2f} m2")

if 250 > dien_tich_max or 250 < dien_tich_min:
    print("Can nha 250 m2 nam ngoai pham vi du lieu.")
    print("Day la truong hop ngoai suy.")
else:
    print("Can nha 250 m2 nam trong pham vi du lieu.")

# Lưu kết quả
ket_qua = pd.DataFrame({
    "mo_hinh": ["Hoi quy tuyen tinh", "MLP"],
    "MAE_tap_kiem_tra": [mae_tuyen_tinh, mae_mlp],
    "gia_du_doan_nha_250m2": [gia_tuyen_tinh, gia_mlp]
})

ket_qua.to_csv(
    "outputs/bai6_ket_qua.csv",
    index=False
)

# Lưu thông tin và nhận xét
with open("outputs/bai6_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write("KET QUA HOI QUY GIA NHA\n")
    f.write(f"MAE hoi quy tuyen tinh: {mae_tuyen_tinh:.4f}\n")
    f.write(f"MAE MLP: {mae_mlp:.4f}\n\n")
    f.write("DU DOAN CAN NHA 250 M2\n")
    f.write(f"So phong: {nha_moi['so_phong'].iloc[0]}\n")
    f.write(f"Tuoi nha: {nha_moi['tuoi_nha'].iloc[0]}\n")
    f.write(f"Gia hoi quy tuyen tinh: {gia_tuyen_tinh:.4f}\n")
    f.write(f"Gia MLP: {gia_mlp:.4f}\n\n")
    f.write(f"Pham vi dien tich: {dien_tich_min:.2f} - {dien_tich_max:.2f} m2\n")

    if 250 > dien_tich_max or 250 < dien_tich_min:
        f.write("Can nha 250 m2 nam ngoai pham vi du lieu.\n")
        f.write("Day la truong hop ngoai suy.\n")

