'''
Bài tập 4: Thêm số phòng vào mô hình
Huấn luyện một mô hình hồi quy tuyến tính bằng scikit-learn với hai cột đầu vào là
dien_tich và so_phong. Chia dữ liệu thành tập học và tập kiểm tra rồi in ra R2
trên tập
kiểm tra. So sánh con số của sinh viên với R2 = 0.9622 của mô hình một biến trong mục 6
và nói xem thêm cột này có đáng hay không.
Bài này dùng train_test_split, LinearRegression và r2_score, cả ba đều đã xuất hiện
ở mục 6 và mục 8. Hãy giữ nguyên test_size=0.2 và random_state=42 như trong tài
liệu. Đó là điều kiện để sinh viên vẫn chia được 48 căn để học và 12 căn để kiểm tra đúng
như tài liệu, nhờ vậy phép so sánh mới công bằng. Gợi ý, so với tệp b7_nhieu_bien.py
sinh viên chỉ cần sửa danh sách cột bên trong hai cặp ngoặc vuông của X. Nộp tệp bai4.py,
ảnh chụp kết quả và câu so sánh.
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv("data/gia_nha.csv")

# Chọn hai biến đầu vào và biến cần dự đoán
X = df[["dien_tich", "so_phong"]]
y = df["gia"]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_tr, y_tr)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_te)
r2 = r2_score(y_te, y_pred)

print("KET QUA HOI QUY TUYEN TINH DA BIEN")
print("So mau huan luyen:", len(X_tr))
print("So mau kiem tra:", len(X_te))
print(f"R2 tren tap kiem tra: {r2:.4f}")

# So sánh với mô hình một biến trong tài liệu
r2_mot_bien = 0.9622

print()
print(f"R2 mo hinh mot bien: {r2_mot_bien:.4f}")
print(f"R2 mo hinh hai bien: {r2:.4f}")

chenh_lech = r2 - r2_mot_bien

if chenh_lech > 0:
    print(f"R2 tang {chenh_lech:.4f} khi them so phong.")
    print(
        "Nhan xet: Them bien so_phong giup mo hinh "
        "giai thich du lieu tot hon tren tap kiem tra."
    )
elif chenh_lech < 0:
    print(f"R2 giam {abs(chenh_lech):.4f} khi them so phong.")
    print(
        "Nhan xet: Them bien so_phong chua giup mo hinh "
        "du doan tot hon tren tap kiem tra."
    )
else:
    print("R2 khong thay doi khi them so phong.")
    print(
        "Nhan xet: Them bien so_phong khong tao ra "
        "su cai thien ve R2 tren tap kiem tra."
    )
