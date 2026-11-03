# -*- coding: utf-8 -*-

"""Bai tap 3: Danh gia anh huong cua he so lam tron alpha den chi so danh gia."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

danh_sach_alpha = [0.001, 0.1, 1.0, 10.0, 100.0]

print(f"{'Alpha':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 46)

for alpha in danh_sach_alpha:
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"{alpha:<10.3f} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f}")

print(
    "\n[NHAN XET]: Khi alpha qua lon (vd: 100.0), cac xac suat cua tu bi keo ve gan"
    " nhu nhau (lam tron qua muc), khien mo hinh mat di kha nang phan biet dac trung"
    " giua cac tu khoa, dan den hieu suat phan loai giam sut nghiem trong."
)
print("=" * 60)