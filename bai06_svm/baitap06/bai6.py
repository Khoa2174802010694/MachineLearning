'''
Bài tập 6: SVM trên dữ liệu lớn
Tạo dữ liệu bằng make_classification(n_samples=n, n_features=20,
random_state=0) với n bằng 2000, 5000, 10000 và 20000. Với mỗi cỡ, đo thời
gian .fit của SVC() và của LinearSVC() bằng mô đun time.
Lập bảng rồi nhận xét: khi dữ liệu tăng gấp đôi, thời gian học của mỗi mô hình tăng khoảng
bao nhiêu lần? Kết quả thời gian khác nhau giữa các máy là chuyện bình thường, chỉ cần
so tương đối.
'''
import time

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.svm import SVC, LinearSVC


kich_thuoc = [2000, 5000, 10000, 20000]
ket_qua = []

for n in kich_thuoc:
    X, y = make_classification(
        n_samples=n,
        n_features=20,
        random_state=0
    )

    svm = SVC()
    bat_dau = time.perf_counter()
    svm.fit(X, y)
    thoi_gian_svc = time.perf_counter() - bat_dau

    linear_svm = LinearSVC(random_state=0, max_iter=10000)
    bat_dau = time.perf_counter()
    linear_svm.fit(X, y)
    thoi_gian_linear = time.perf_counter() - bat_dau

    ket_qua.append({
        "So mau": n,
        "SVC (giay)": thoi_gian_svc,
        "LinearSVC (giay)": thoi_gian_linear
    })

    print(f"So mau: {n}")
    print(f"SVC: {thoi_gian_svc:.4f} giay")
    print(f"LinearSVC: {thoi_gian_linear:.4f} giay")
    print()

bang = pd.DataFrame(ket_qua)

print("Bang so sanh thoi gian:")
print(bang.to_string(index=False))

print()
print("Ty le tang thoi gian khi kich thuoc du lieu tang:")

for i in range(1, len(ket_qua)):
    ty_le_svc = (
        ket_qua[i]["SVC (giay)"]
        / ket_qua[i - 1]["SVC (giay)"]
    )

    ty_le_linear = (
        ket_qua[i]["LinearSVC (giay)"]
        / ket_qua[i - 1]["LinearSVC (giay)"]
    )

    print(
        f"{kich_thuoc[i - 1]} -> {kich_thuoc[i]} mau: "
        f"SVC = {ty_le_svc:.2f} lan, "
        f"LinearSVC = {ty_le_linear:.2f} lan"
    )
