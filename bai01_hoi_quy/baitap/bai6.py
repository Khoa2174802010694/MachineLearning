import pandas as pd

df = pd.read_csv("bai01_hoi_quy/data/gia_nha.csv")

w = 0.078367
b = 0.401752

def du_doan_gia(dien_tich):
    if dien_tich < 35.5 or dien_tich > 117.5:
        print("Canh bao: Dien tich nam ngoai pham vi du lieu.")

    return w * dien_tich + b

for dien_tich in [60, 80, 200]:
    gia = du_doan_gia(dien_tich)
    print(f"{dien_tich} m2 -> {gia:.3f} ty dong")