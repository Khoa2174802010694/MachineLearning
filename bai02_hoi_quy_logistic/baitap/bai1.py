# -*- coding: utf-8 -*-

"""
Bài tập 1: Thống kê theo nhóm.

So sánh tỷ lệ qua môn giữa:
- Nhóm có điểm giữa kỳ từ 7 trở lên.
- Nhóm còn lại.
"""

import pandas as pd


# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("data/sinh_vien.csv")


# Chia sinh viên thành hai nhóm theo điểm giữa kỳ
nhom_cao = df[df["diem_giua_ky"] >= 7]
nhom_con_lai = df[df["diem_giua_ky"] < 7]


# Tính các số liệu cần thiết
so_ban_nhom_cao = len(nhom_cao)
ty_le_qua_nhom_cao = nhom_cao["qua_mon"].mean()
ty_le_qua_nhom_con_lai = nhom_con_lai["qua_mon"].mean()


# In kết quả
print("So ban co diem giua ky tu 7 tro len:", so_ban_nhom_cao)
print(
    "Ty le qua mon cua nhom diem tu 7:",
    f"{ty_le_qua_nhom_cao:.4f}",
)
print(
    "Ty le qua mon cua nhom con lai:",
    f"{ty_le_qua_nhom_con_lai:.4f}",
)
print()


# Nhận xét
print("Nhan xet:")

if ty_le_qua_nhom_cao > ty_le_qua_nhom_con_lai:
    print(
        "Nhom co diem giua ky tu 7 tro len co ty le qua mon cao hon, "
        "cho thay diem giua ky co kha nang phan biet hai nhom."
    )
else:
    print(
        "Ty le qua mon cua hai nhom khong cho thay diem giua ky "
        "phan biet ro hai nhom."
    )