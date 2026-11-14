'''
Bài tập 1: Tính tay khoảng cách tới ranh giới
Dùng w1, w2, b in ra ở mục 2, tự tính f(x) cho hai điểm (3, 3) và (5, 4.5) bằng phép nhân
và cộng, không dùng decision_function.
Khoảng cách từ một điểm tới đường ranh giới bằng |f(x)|/√w1^2 + w2^2. Tính khoảng cách 
của hai điểm trên, và của ba vectơ hỗ trợ. Vì sao khoảng cách của ba vectơ hỗ trợ bằng đúng
một nửa độ rộng lề?
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

svm = SVC(kernel="linear", C=1000)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

do_dai_w = np.linalg.norm(w)
do_rong_le = 2 / do_dai_w

print(f"w1 = {w[0]:.4f}, w2 = {w[1]:.4f}, b = {b:.4f}")
print(f"Do rong le = {do_rong_le:.4f}")
print()

diem_moi = np.array([
    [3.0, 3.0],
    [5.0, 4.5]
])

print("Khoang cach cua hai diem moi:")

for diem in diem_moi:
    f = w[0] * diem[0] + w[1] * diem[1] + b
    khoang_cach = abs(f) / do_dai_w

    print(
        f"Diem ({diem[0]:.1f}, {diem[1]:.1f}): "
        f"f(x) = {f:.4f}, khoang cach = {khoang_cach:.4f}"
    )

print()

print("Khoang cach cua cac vecto ho tro:")

for i in svm.support_:
    diem = X[i]

    f = w[0] * diem[0] + w[1] * diem[1] + b
    khoang_cach = abs(f) / do_dai_w

    print(
        f"Diem {i:2d}: "
        f"x1 = {diem[0]:.2f}, x2 = {diem[1]:.2f}, "
        f"lop = {y[i]}, khoang cach = {khoang_cach:.4f}"
    )

print()
print(f"Mot nua do rong le = {do_rong_le / 2:.4f}")
