# -*- coding: utf-8 -*-

"""Bai tap 1: Mo rong dinh ly Bayes cho truong hop ba nguyen nhan.

Giai thich: Khi co nhieu hon hai gia thuyet (nhom), mau so cua dinh ly Bayes
se la tong tich cua xac suat tien nghiem va kha nang cua tat ca cac nhom.
Tong cac xac suat hau nghiem sau khi tinh phai luon dung bang 1.0.
"""


def bayes_nhieu_nhanh(tien_nghiem_list, kha_nang_list):
    """Tinh xac suat hau nghiem cho danh sách các nguyen nhan."""
    tu_sos = [kha_nang * tien for tien, kha_nang in zip(tien_nghiem_list, kha_nang_list)]
    mau_so = sum(tu_sos)
    hau_nghiem = [tu / mau_so for tu in tu_sos]
    return hau_nghiem


# Du lieu dau vao: [Nhom hoc deu, Nhom hoc don, Nhom it học]
xac_suat_tien_nghiem = [0.50, 0.30, 0.20]
kha_nang_qua_mon = [0.95, 0.70, 0.30]
ten_cac_nhom = ["Nhom hoc deu", "Nhom hoc don", "Nhom it hoc"]

ket_qua_hau_nghiem = bayes_nhieu_nhanh(xac_suat_tien_nghiem, kha_nang_qua_mon)

print("Xac suat hau nghiem khi biet sinh vien da qua mon:")
for ten_nhom, xac_suat in zip(ten_cac_nhom, ket_qua_hau_nghiem):
    print(f" - {ten_nhom:15s} = {xac_suat:.4f} ({xac_suat * 100:.2f}%)")

tong_kiem_tra = sum(ket_qua_hau_nghiem)
print("-" * 45)
print(f"Tong xac suat kiem tra = {tong_kiem_tra:.4f}")
print(
    "\n[NHAN XET]: Mac du ban dau nhom 'hoc deu' chiem 50%, khi biet chac sinh vien"
    " da qua mon, xac suat ban do thuoc nhom hoc deu tang vot len ~63.76% do kha nang"
    " qua mon cua nhom nay rat cao (95%). Tong xac suat dat dung 1.0."
)
print("=" * 60)