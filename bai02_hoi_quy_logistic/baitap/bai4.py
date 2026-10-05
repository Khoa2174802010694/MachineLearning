# -*- coding: utf-8 -*-

"""
Bài tập 4: Tự tính Accuracy, Precision, Recall và F1.

Các chỉ số được tính trực tiếp từ:
TP, TN, FP, FN.
"""

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Đọc dữ liệu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)


# Huấn luyện mô hình
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Dự đoán trên tập kiểm tra
y_pred = mo_hinh.predict(X_test)


# Lấy TP, TN, FP, FN từ ma trận nhầm lẫn
tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
).ravel()


print("Cac gia tri tu ma tran nham lan:")
print(f"TN = {tn}")
print(f"FP = {fp}")
print(f"FN = {fn}")
print(f"TP = {tp}")
print()


# Tự tính Accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)


# Tự tính Precision
precision = tp / (tp + fp)


# Tự tính Recall
recall = tp / (tp + fn)


# Tự tính F1
f1 = 2 * precision * recall / (precision + recall)


# In kết quả
print("Ket qua tu tinh:")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")
print()


# In công thức để dễ kiểm tra
print("Cong thuc:")
print(
    f"Accuracy  = ({tp} + {tn}) / "
    f"({tp} + {tn} + {fp} + {fn})"
)
print(
    f"Precision = {tp} / ({tp} + {fp})"
)
print(
    f"Recall    = {tp} / ({tp} + {fn})"
)
print(
    f"F1        = 2 * Precision * Recall / "
    f"(Precision + Recall)"
)