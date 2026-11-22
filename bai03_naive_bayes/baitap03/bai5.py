'''
Bài tập 5: Chọn sai biến thể Naive Bayes
Chạy MultinomialNB trên bộ dữ liệu sinh viên với hai cột gio_on và diem_giua_ky. In ra
accuracy rồi so với GaussianNB ở mục 6.
Chương trình có báo lỗi không? Con số accuracy ra bao nhiêu? Giải thích vì sao dùng
MultinomialNB ở đây là sai về nguyên tắc, dù nó vẫn chạy
'''
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sinh_vien.csv"
OUTPUT_DIR = ROOT / "outputs"

df = pd.read_csv(DATA_PATH)

X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# MultinomialNB van chay vi cac gia tri dau vao khong am.
model_multi = MultinomialNB()
model_multi.fit(X_train, y_train)
pred_multi = model_multi.predict(X_test)
acc_multi = accuracy_score(y_test, pred_multi)

# GaussianNB phu hop hon voi cac dac trung lien tuc.
model_gaussian = GaussianNB()
model_gaussian.fit(X_train, y_train)
pred_gaussian = model_gaussian.predict(X_test)
acc_gaussian = accuracy_score(y_test, pred_gaussian)

print("MultinomialNB accuracy:", f"{acc_multi:.4f}")
print("GaussianNB accuracy:", f"{acc_gaussian:.4f}")

ket_qua = pd.DataFrame({
    "mo_hinh": ["MultinomialNB", "GaussianNB"],
    "accuracy": [acc_multi, acc_gaussian]
})

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ket_qua.to_csv(OUTPUT_DIR / "bai5_so_sanh.csv", index=False, encoding="utf-8-sig")

chenh_lech = acc_gaussian - acc_multi
nhan_xet = (
    "Bai tap 5 - Chon sai bien the Naive Bayes\n"
    f"So mau train: {len(X_train)}\n"
    f"So mau test: {len(X_test)}\n"
    f"MultinomialNB accuracy: {acc_multi:.4f}\n"
    f"GaussianNB accuracy: {acc_gaussian:.4f}\n"
    f"Chenh lech (GaussianNB - MultinomialNB): {chenh_lech:.4f}\n\n"
    "MultinomialNB khong bao loi trong bai nay vi cac gia tri gio_on va "
    "diem_giua_ky deu khong am. Tuy nhien, mo hinh nay duoc thiet ke cho "
    "du lieu dang dem/tan suat va mo hinh hoa xac suat theo tong gia tri "
    "dac trung. Hai cot gio_on va diem_giua_ky la bien so lien tuc, khong "
    "phai so lan xuat hien. GaussianNB phu hop hon vi no mo hinh hoa moi "
    "dac trung lien tuc bang phan phoi Gaussian trong tung lop. Accuracy "
    "cao hon khong lam thay doi su khong phu hop ve gia dinh cua MultinomialNB."
)
(OUTPUT_DIR / "bai5_nhan_xet.txt").write_text(nhan_xet, encoding="utf-8")
