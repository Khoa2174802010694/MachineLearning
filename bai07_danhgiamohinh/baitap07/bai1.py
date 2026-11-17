'''
Bài tập 1: Tự viết hàm đánh giá
Viết hàm danh_gia(y_that, y_doan) trả về accuracy, precision, recall và F1, chỉ dùng
numpy. Chạy hàm cho mô hình logistic và cho mô hình lười ở mục 2, so với các hàm của
scikit-learn.
Với mô hình lười, precision có tính được không? Hàm của em xử lý trường hợp chia cho 0
thế nào?
'''
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)

    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))

    tong = tp + fp + fn + tn

    accuracy = (tp + tn) / tong if tong > 0 else 0
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return accuracy, precision, recall, f1


df = pd.read_csv("data/sinh_vien_bo_hoc.csv")

X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

mo_hinh = {
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
    "Logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression()
    )
}

for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)

    kq = danh_gia(y_te, du)

    acc = accuracy_score(y_te, du)
    pre = precision_score(y_te, du, zero_division=0)
    rec = recall_score(y_te, du, zero_division=0)
    f1 = f1_score(y_te, du, zero_division=0)

    print(f"\n{ten}")
    print("Tu viet:")
    print(
        f"Accuracy = {kq[0]:.4f}, "
        f"Precision = {kq[1]:.4f}, "
        f"Recall = {kq[2]:.4f}, "
        f"F1 = {kq[3]:.4f}"
    )
    print("Scikit-learn:")
    print(
        f"Accuracy = {acc:.4f}, "
        f"Precision = {pre:.4f}, "
        f"Recall = {rec:.4f}, "
        f"F1 = {f1:.4f}"
    )

'''
Với mô hình lười, tất cả dự đoán đều thuộc lớp 0 nên TP = 0 và FP = 0. 
Precision có mẫu số bằng 0, do đó không xác định theo công thức toán học. 
Trong chương trình, quy ước trả về 0 để tránh lỗi chia cho 0 và thống nhất với Scikit-learn khi dùng zero_division=0.
'''