'''
Bài tập 1: Lọc ra nhóm căn hộ lớn
Đọc bộ dữ liệu, sau đó in ra hai con số. Con số thứ nhất là số căn có diện tích lớn hơn 100
mét vuông. Con số thứ hai là giá trung bình của riêng nhóm căn đó, tính bằng tỷ đồng.
Bài này chỉ cần các hàm đã dùng ở mục 3, đó là pd.read_csv để đọc tệp và mean để lấy
trung bình. Gợi ý, pandas cho phép lọc bằng một điều kiện đúng sai (boolean mask), sinh
viên viết nhom_lon = df[df["dien_tich"] > 100] rồi đếm bằng len(nhom_lon). Giá
trung bình của nhóm lấy bằng nhom_lon["gia"].mean(). Toàn bộ 60 căn có diện tích trải
từ 35.5 tới 117.5 mét vuông, nên nhóm sinh viên lọc ra sẽ nhỏ hơn 60 khá nhiều.
'''
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

# Lọc các căn hộ có diện tích lớn hơn 100 m2
nhom_lon = df[df["dien_tich"] > 100]

so_can = len(nhom_lon)
gia_trung_binh = nhom_lon["gia"].mean()

print("KET QUA LOC CAN HO LON")
print("So can ho co dien tich tren 100 m2:", so_can)
print(f"Gia trung binh cua nhom: {gia_trung_binh:.4f} ty dong")

print()
print(
    f"Nhan xet: Co {so_can} can ho co dien tich tren 100 m2, "
    f"voi gia trung binh {gia_trung_binh:.4f} ty dong."
)
