# -*- coding: utf-8 -*-

"""Bai tap 2: Tim tu to cao tin rac va tin thuong manh nhat dua tren feature_log_prob_."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data/tin_nhan.csv")
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

mo_hinh = MultinomialNB(alpha=1.0)
mo_hinh.fit(X, y)

# Lay danh sach tu vung va log-probabilities cua 2 lop (lop 0: thuong, lop 1: rac)
tu_vung = vec.get_feature_names_out()
log_prob_rac = mo_hinh.feature_log_prob_[1]
log_prob_thuong = mo_hinh.feature_log_prob_[0]

# Do chech lech log_prob_rac - log_prob_thuong cang lon -> tu do cang thien ve tin rac
bang_diem_tu = pd.DataFrame(
    {
        "tu": tu_vung,
        "chenh_lech": log_prob_rac - log_prob_thuong,
    }
)

print("5 tu to cao tin RAC manh nhat:")
top_rac = bang_diem_tu.sort_values("chenh_lech", ascending=False).head(5)
for _, row in top_rac.iterrows():
    print(f" - {row['tu']:15s} (diem chenh lech: {row['chenh_lech']:.4f})")
print()

print("5 tu cho thay tin THUONG manh nhat:")
top_thuong = bang_diem_tu.sort_values("chenh_lech", ascending=True).head(5)
for _, row in top_thuong.iterrows():
    print(f" - {row['tu']:15s} (diem chenh lech: {row['chenh_lech']:.4f})")

print(
    "\n[NHAN XET]: Su dung hieu so log_prob giup xac dinh tu nao co xac suat xuat hien"
    " o lop nay vuot troi hoan toan so voi lop kia, qua do phan anh chinh xac tu khoa"
    " dac trung cua tin rac (vd: khuyen mai, trung thuong) hoac tin thuong (vd: nhom, bai tap)."
)
print("=" * 60)