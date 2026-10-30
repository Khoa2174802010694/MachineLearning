# -*- coding: utf-8 -*-

"""Bai tap 4: Tu cai dat Naive Bayes thu cong tren toan bo tap du lieu (dung log).

Giai thich co che:
Thay vi nhan nhieu xac suat nho (de gay tran so - underflow), ta dung phep cong
cac logarit: log(P) = log(P1) + log(P2) + ...
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X_sparse = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.25, random_state=17, stratify=y
)

# Chuyen doi sang ma tran mat do de tien thao tac thu cong
X_train_dense = X_train.toarray()
X_test_dense = X_test.toarray()
tu_vung = vec.get_feature_names_out()
V = len(tu_vung)


class CustomNaiveBayesLog:

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.log_prior = {}
        self.log_likelihood = {}

    def fit(self, X, y):
        tong_so_mau = len(y)
        cac_lop = np.unique(y)

        for c in cac_lop:
            X_c = X[y == c]
            # Xac suat tien nghiem log P(C)
            self.log_prior[c] = np.log(len(X_c) / tong_so_mau)

            # Tong so lan xuat hien cua cac tu trong lop c
            tong_tu_trong_lop = X_c.sum(axis=0)
            tong_tong_tu = tong_tu_trong_lop.sum()

            # Tinh log likelihood co ap dung Laplace Smoothing
            self.log_likelihood[c] = np.log(
                (tong_tu_trong_lop + self.alpha) / (tong_tong_tu + self.alpha * V)
            )

    def predict_log_proba(self, X):
        ket_qua = []
        for row in X:
            diem_cac_lop = {}
            for c in self.log_prior:
                # Tong loga tien nghiem + tong loga xac suat cac tu xuat hien trong cau
                diem = self.log_prior[c] + np.sum(row * self.log_likelihood[c])
                diem_cac_lop[c] = diem
            ket_qua.append(diem_cac_lop)
        return ket_qua

    def predict(self, X):
        diem_list = self.predict_log_proba(X)
        return [max(d, key=d.get) for d in diem_list]


# 1. Chay mo hình tu viet
my_nb = CustomNaiveBayesLog(alpha=1.0)
my_nb.fit(X_train_dense, y_train)
y_pred_my = my_nb.predict(X_test_dense)
acc_my = accuracy_score(y_test, y_pred_my)

# 2. Chay mo hinh chuan tu scikit-learn de doi chieu
sk_nb = MultinomialNB(alpha=1.0)
sk_nb.fit(X_train, y_train)
y_pred_sk = sk_nb.predict(X_test)
acc_sk = accuracy_score(y_test, y_pred_sk)

print(f"Accuracy bo loc tu viet : {acc_my:.4f}")
print(f"Accuracy cua scikit-learn: {acc_sk:.4f}")
print(
    "\n[NHAN XET]: Hai ket qua accuracy trung khop hoac rat gan nhau, chung to code"
    " tu viet da hieu dung ban chất toan hoc (tinh log-likelihood va ap dung Laplace Smoothing)."
)
print("=" * 60)