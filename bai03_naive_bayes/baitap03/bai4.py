'''
Bài tập 4: Tự viết bộ lọc không dùng thư viện
Dựa vào d4_lam_tron.py, viết một bộ lọc hoàn chỉnh cho toàn bộ 300 tin nhắn. Tự đếm
từ, tự làm trơn, tự tính điểm cho hai lớp bằng tổng loga thay vì tích.
Chia dữ liệu bằng train_test_split với random_state=17 rồi so accuracy của bộ lọc tự
viết với MultinomialNB. Hai con số phải rất gần nhau. Nếu lệch nhiều thì có lỗi ở đâu đó,
hãy tìm ra.
'''
import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Doc du lieu
df = pd.read_csv("data/tin_nhan.csv")
X = df["noi_dung"].fillna("").astype(str)
y = df["la_rac"].astype(int)

# Chia train/test co dinh de hai mo hinh dung cung mot tap du lieu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Tao tu dien tu tap train, dung cach tach tu nhu file d4_lam_tron.py
tu_dien = sorted({
    tu
    for cau in X_train
    for tu in cau.split()
})
V = len(tu_dien)
tap_tu = set(tu_dien)

# Dem so lan xuat hien cua tung tu theo tung lop
dem = {0: Counter(), 1: Counter()}
tong = {0: 0, 1: 0}
so_mau = {0: 0, 1: 0}

for cau, nhan in zip(X_train, y_train):
    nhan = int(nhan)
    so_mau[nhan] += 1
    for tu in cau.split():
        if tu in tap_tu:
            dem[nhan][tu] += 1
            tong[nhan] += 1

# Laplace smoothing voi alpha = 1.0, giong MultinomialNB mac dinh
alpha = 1.0
tong_mau = len(y_train)
log_tien_nghiem = {
    nhan: np.log(so_mau[nhan] / tong_mau)
    for nhan in (0, 1)
}
log_xac_suat_tu = {
    nhan: {
        tu: np.log(
            (dem[nhan][tu] + alpha) /
            (tong[nhan] + alpha * V)
        )
        for tu in tu_dien
    }
    for nhan in (0, 1)
}

def diem_log(cau, nhan):
    """Tinh tong log xac suat thay cho tich xac suat."""
    diem = log_tien_nghiem[nhan]
    for tu, so_lan in Counter(cau.split()).items():
        if tu in tap_tu:
            diem += so_lan * log_xac_suat_tu[nhan][tu]
    return diem

def du_doan_tu_viet(cau):
    diem_rac = diem_log(cau, 1)
    diem_thuong = diem_log(cau, 0)
    return 1 if diem_rac > diem_thuong else 0

y_pred_tu_viet = np.array([du_doan_tu_viet(cau) for cau in X_test])
accuracy_tu_viet = accuracy_score(y_test, y_pred_tu_viet)

# Mo hinh sklearn dung cung cach tach tu va cung tap train/test
vectorizer = CountVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

mo_hinh = MultinomialNB(alpha=alpha)
mo_hinh.fit(X_train_vec, y_train)
y_pred_sklearn = mo_hinh.predict(X_test_vec)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

# Luu ket qua
os.makedirs("outputs", exist_ok=True)

bang_ket_qua = pd.DataFrame({
    "Mo_hinh": ["Bo loc tu viet (tong log)", "MultinomialNB"],
    "Accuracy": [accuracy_tu_viet, accuracy_sklearn]
})
bang_ket_qua.to_csv(
    "outputs/bai4_so_sanh.csv", index=False, encoding="utf-8-sig"
)

chenh_lech = abs(accuracy_tu_viet - accuracy_sklearn)
noi_dung = (
    "BAI 4 - SO SANH BO LOC NAIVE BAYES\n"
    f"So mau train: {len(X_train)}\n"
    f"So mau test: {len(X_test)}\n"
    f"So tu trong tu dien: {V}\n"
    f"Alpha: {alpha}\n"
    f"Accuracy bo loc tu viet: {accuracy_tu_viet:.4f}\n"
    f"Accuracy MultinomialNB: {accuracy_sklearn:.4f}\n"
    f"Chenh lech accuracy: {chenh_lech:.4f}\n"
    "\nNhan xet:\n"
    "- Bo loc tu viet dem tan suat tu theo tung lop va ap dung lam tron Laplace.\n"
    "- Diem moi lop duoc tinh bang tong log xac suat, tranh tran so khi nhan nhieu so nho.\n"
    "- Hai mo hinh su dung cung tap train/test, tu dien, cach tach tu va alpha.\n"
    "- Neu cai dat tu viet dung cung cong thuc MultinomialNB, accuracy se trung khop "
    "hoac rat gan nhau.\n"
)
with open("outputs/bai4_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write(noi_dung)

print("KET QUA BAI 4")
print(f"So tu trong tu dien: {V}")
print(f"Accuracy bo loc tu viet: {accuracy_tu_viet:.4f}")
print(f"Accuracy MultinomialNB: {accuracy_sklearn:.4f}")
print(f"Chenh lech: {chenh_lech:.4f}")
print("\nBang so sanh:")
print(bang_ket_qua.to_string(index=False))
