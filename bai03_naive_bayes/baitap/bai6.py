# -*- coding: utf-8 -*-

"""Bai tap 6: Kho du lieu mat can bang nang va cai bay Accuracy."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")

# Tao tap du lieu mat can bang: giu toan bo 180 tin thuong (la_rac == 0) va chi lay 20 tin rac dau tien (la_rac == 1)
df_thuong = df[df["la_rac"] == 0]
df_rac = df[df["la_rac"] == 1].head(20)
df_mat_can_bang = pd.concat([df_thuong, df_rac]).reset_index(drop=True)

X = vec = CountVectorizer(token_pattern=r"(?u)\S+").fit_transform(df_mat_can_bang["noi_dung"])
y = df_mat_can_bang["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

acc = accuracy_score(y_test, y_pred)
rec_rac = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

print(f"Accuracy tren tap mat can bang: {acc:.4f}")
print(f"Recall cua lop rac (pos_label=1): {rec_rac:.4f}")
print(
    "\n[NHAN XET & GIAI THICH]:\n"
    "- Accuracy van rat cao (vi chiem phan lon la tin thuong), nhung do do do hoan toan"
    "  DAH LUA nguoi dung (Bay trau bo / Accuracy paradox).\n"
    "- Chi so Recall cua lop rac phan anh thuc chat kha nang bat tin rac cua mo hinh"
    "  se sut giam nghiem trọng do thieu du lieu dai dien o lop rac."
)