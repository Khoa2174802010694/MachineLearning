# -*- coding: utf-8 -*-

"""Bai tap 5: Chay MultinomialNB tren du lieu lien tuc (gio_on, diem_giua_ky)."""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

df = pd.read_csv("data/sinh_vien.csv")
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Su dung sai bien the: MultinomialNB cho du lieu lien tuc
multi_nb = MultinomialNB()
multi_nb.fit(X_train, y_train)
acc_multi = accuracy_score(y_test, multi_nb.predict(X_test))

# Su dung dung bien the: GaussianNB
gauss_nb = GaussianNB()
gauss_nb.fit(X_train, y_train)
acc_gauss = accuracy_score(y_test, gauss_nb.predict(X_test))

print(f"Accuracy dung MultinomialNB: {acc_multi:.4f}")
print(f"Accuracy dung GaussianNB   : {acc_gauss:.4f}")
print(
    "\n[NHAN XET & GIAI THICH]:\n"
    "1. Chuong trinh KHONG bao loi vi sklearn van chap nhan mang so.\n"
    "2. Tuy nhien, MultinomialNB duoc thiet kế cho du lieu ROI RAC (dem so lan xuat hien, tan so).\n"
    "   Trong khi do, gio_on va diem_giua_ky la bien SO LIEU LIEN TUC. Việc dung MultinomialNB"
    "   lam sai ban chat phan phoi xac suat, dan den hieu suat du doan kem hieu qua hon GaussianNB"
    "   (bien the dung de xu ly phan phoi chuan lien tuc)."
)
print("=" * 60)