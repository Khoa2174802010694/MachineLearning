'''
Bài tập 1: Bayes với ba nguyên nhân
Một lớp có ba nhóm sinh viên: 50 phần trăm học đều, 30 phần trăm học dồn, 20 phần trăm
ít học. Tỷ lệ qua môn của ba nhóm lần lượt là 0.95, 0.70 và 0.30.
Gặp một bạn đã qua môn. Tính xác suất bạn đó thuộc từng nhóm. Ba con số cộng lại phải
bằng 1, hãy dùng điều đó để tự kiểm tra.
Gợi ý, mở rộng hàm bayes trong d1_bayes_tay.py cho ba nhánh thay vì hai.
'''
def bayes_ba_nhom(tien_nghiem, kha_nang):

    tu_so = [
        tien_nghiem[i] * kha_nang[i]
        for i in range(len(tien_nghiem))
    ]

    mau_so = sum(tu_so)

    hau_nghiem = [
        gia_tri / mau_so
        for gia_tri in tu_so
    ]

    return hau_nghiem


# Xac suat tien nghiem cua ba nhom
tien_nghiem = [0.50, 0.30, 0.20]

# Xac suat qua mon cua tung nhom
kha_nang_qua_mon = [0.95, 0.70, 0.30]

# Tinh xac suat hau nghiem khi biet sinh vien da qua mon
ket_qua = bayes_ba_nhom(tien_nghiem, kha_nang_qua_mon)

ten_nhom = ["Hoc deu", "Hoc don", "It hoc"]

print("BAYES VOI BA NGUYEN NHAN")
print("=" * 42)

for ten, xac_suat in zip(ten_nhom, ket_qua):
    print(f"Xac suat thuoc nhom {ten:<10}: {xac_suat:.4f} ({xac_suat:.2%})")

# Kiem tra tong cac xac suat hau nghiem
tong_hau_nghiem = sum(ket_qua)

print("-" * 42)
print(f"Tong ba xac suat hau nghiem: {tong_hau_nghiem:.4f}")

if abs(tong_hau_nghiem - 1.0) < 1e-9:
    print("Kiem tra: Hop le, tong bang 1.")
else:
    print("Kiem tra: Khong hop le, tong khac 1.")