'''
Bài tập 5: Dò ngưỡng tốt nhất theo F1
Cho ngưỡng chạy từ 0.05 tới 0.95, mỗi bước 0.05. Với mỗi ngưỡng, tính F1 trên tập kiểm
tra rồi in ra thành bảng. Cuối cùng in ra ngưỡng cho F1 cao nhất.
Nhận xét một câu: ngưỡng tốt nhất theo F1 có đúng bằng 0.5 không? Gợi ý, dùng
np.arange(0.05, 1.0, 0.05) và hàm f1_score với tham số zero_division=0.
'''
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia du lieu thanh tap huan luyen va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)


# Huan luyen mo hinh
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Lay xac suat qua mon tren tap kiem tra
xac_suat = mo_hinh.predict_proba(X_test)[:, 1]


# Thu cac nguong tu 0.05 den 0.95
cac_nguong = np.arange(0.05, 1.0, 0.05)

ket_qua = []

for nguong in cac_nguong:
    y_pred = (xac_suat >= nguong).astype(int)

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    ket_qua.append((nguong, f1))


# Tim nguong co F1 cao nhat
nguong_tot_nhat, f1_cao_nhat = max(
    ket_qua,
    key=lambda item: item[1]
)


# In bang ket qua
print("KET QUA DO NGUONG THEO F1")
print("=" * 35)
print(f"{'Nguong':<15}{'F1':>10}")
print("-" * 25)

for nguong, f1 in ket_qua:
    print(f"{nguong:<15.2f}{f1:>10.4f}")

print("-" * 25)
print(f"Nguong tot nhat: {nguong_tot_nhat:.2f}")
print(f"F1 cao nhat:     {f1_cao_nhat:.4f}")


# Nhan xet
print("\nNHAN XET:")
if np.isclose(nguong_tot_nhat, 0.5):
    print("Nguong toi uu theo F1 trung voi nguong mac dinh 0.5.")
else:
    print(
        f"Nguong toi uu theo F1 la {nguong_tot_nhat:.2f}, "
        "khac nguong mac dinh 0.5."
    )