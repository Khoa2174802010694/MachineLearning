'''
Bài tập 1: Thống kê theo nhóm
Đọc bộ dữ liệu rồi in ra ba con số. Số bạn có điểm giữa kỳ từ 7 trở lên, tỷ lệ qua môn của
riêng nhóm đó, và tỷ lệ qua môn của nhóm còn lại.
Gợi ý, dùng phép lọc theo điều kiện của pandas rồi lấy .mean() của cột qua_mon. Nhận xét
một câu về việc điểm giữa kỳ có phân biệt được hai nhóm hay không
'''
import pandas as pd

# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

# Chia sinh vien thanh hai nhom theo diem giua ky
nhom_diem_cao = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]

# Tinh cac chi so thong ke
so_sinh_vien = len(nhom_diem_cao)
ty_le_qua_mon_nhom_cao = nhom_diem_cao["qua_mon"].mean()
ty_le_qua_mon_nhom_con_lai = nhom_con_lai["qua_mon"].mean()

# In ket qua
print("THONG KE THEO NHOM DIEM GIUA KY")
print("=" * 45)
print(f"So sinh vien co diem giua ky tu 7 tro len: {so_sinh_vien}")
print(f"Ty le qua mon cua nhom diem tu 7: {ty_le_qua_mon_nhom_cao:.2%}")
print(f"Ty le qua mon cua nhom con lai: {ty_le_qua_mon_nhom_con_lai:.2%}")

# Nhan xet
print("\nNHAN XET")
if ty_le_qua_mon_nhom_cao > ty_le_qua_mon_nhom_con_lai:
    print("Diem giua ky co kha nang phan biet hai nhom,")
    print("vi nhom co diem tu 7 tro len co ty le qua mon cao hon.")
elif ty_le_qua_mon_nhom_cao < ty_le_qua_mon_nhom_con_lai:
    print("Diem giua ky chua phan biet tot hai nhom,")
    print("vi nhom co diem tu 7 tro len co ty le qua mon thap hon.")
else:
    print("Diem giua ky chua the hien kha nang phan biet hai nhom,")
    print("vi hai nhom co ty le qua mon bang nhau.")
