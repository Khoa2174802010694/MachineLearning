'''
Bài tập 2: Thêm một điểm
Thêm vào 24 điểm của mục 2 một điểm lớp 1 tại (-1, -1), học lại và in w1, w2, b. Sau đó
bỏ điểm đó, thay bằng một điểm lớp 1 tại (4.5, 4), học lại và in lần nữa.
Lần nào đường ranh giới thay đổi? Giải thích bằng khái niệm vectơ hỗ trợ.
'''
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC


X, y = make_blobs(
    n_samples=24,
    centers=[[2, 2], [6, 6]],
    cluster_std=0.9,
    random_state=0
)

y = 1 - y


def huan_luyen_va_in(X_data, y_data, ten):
    svm = SVC(kernel="linear", C=1000)
    svm.fit(X_data, y_data)

    w = svm.coef_[0]
    b = svm.intercept_[0]

    print(ten)
    print(f"w1 = {w[0]:.4f}")
    print(f"w2 = {w[1]:.4f}")
    print(f"b = {b:.4f}")
    print("So vecto ho tro:", len(svm.support_))
    print()

    return svm


svm_goc = huan_luyen_va_in(
    X,
    y,
    "Mo hinh ban dau:"
)

diem_1 = np.array([[-1.0, -1.0]])
nhan_1 = np.array([1])

X_them_1 = np.vstack([X, diem_1])
y_them_1 = np.concatenate([y, nhan_1])

svm_them_1 = huan_luyen_va_in(
    X_them_1,
    y_them_1,
    "Them diem lop 1 tai (-1, -1):"
)

diem_2 = np.array([[4.5, 4.0]])

X_them_2 = np.vstack([X, diem_2])
y_them_2 = np.concatenate([y, nhan_1])

svm_them_2 = huan_luyen_va_in(
    X_them_2,
    y_them_2,
    "Them diem lop 1 tai (4.5, 4.0):"
)

print("So sanh voi mo hinh ban dau:")

print(
    "Diem (-1, -1) co lam thay doi ranh gioi:",
    not (
        np.allclose(svm_goc.coef_, svm_them_1.coef_)
        and np.allclose(svm_goc.intercept_, svm_them_1.intercept_)
    )
)

print(
    "Diem (4.5, 4.0) co lam thay doi ranh gioi:",
    not (
        np.allclose(svm_goc.coef_, svm_them_2.coef_)
        and np.allclose(svm_goc.intercept_, svm_them_2.intercept_)
    )
)
