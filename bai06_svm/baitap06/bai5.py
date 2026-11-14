'''
Bài tập 5: Chữ số viết tay, những lỗi nào hay gặp
Dùng confusion_matrix trong sklearn.metrics để in bảng nhầm lẫn 10 hàng 10 cột của
SVM ở mục 6. Hàng là chữ số thật, cột là chữ số đoán.
Đọc bảng và cho biết những cặp chữ số nào bị nhầm. Dùng
plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r") để vẽ và lưu từng ảnh bị
đoán sai, rồi nhận xét vì sao SVM nhầm. Chương 7 sẽ học kỹ về bảng nhầm lẫn.
'''
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


X, y = load_digits(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)

du_doan = svm.predict(X_te)

print(f"Accuracy: {accuracy_score(y_te, du_doan):.4f}")
print()

cm = confusion_matrix(y_te, du_doan)

print("Confusion Matrix:")
print(cm)

sai = np.flatnonzero(du_doan != y_te)

print()
print("So anh doan sai:", len(sai))

os.makedirs("bai06_svm/baitap06/anh_loi", exist_ok=True)

for i in sai:
    anh = X_te[i].reshape(8, 8)

    plt.figure(figsize=(3, 3))
    plt.imshow(anh, cmap="gray_r")
    plt.title(
        f"That: {y_te[i]} - Doan: {du_doan[i]}"
    )
    plt.axis("off")

    ten_file = (
        f"bai06_svm/baitap06/anh_loi/"
        f"anh_{i}_that_{y_te[i]}_doan_{du_doan[i]}.png"
    )

    plt.savefig(
        ten_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

print("Da luu cac anh doan sai.")
