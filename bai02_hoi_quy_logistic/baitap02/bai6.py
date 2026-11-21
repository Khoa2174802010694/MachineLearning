'''
Bài tập 6: Đổi lớp dương rồi chấm lại
Ở mục 5 ta luôn lấy lớp dương là qua môn. Bây giờ hãy chấm điểm cho lớp rớt môn, tức
lớp 0.
Gợi ý, thêm tham số pos_label=0 vào precision_score và recall_score. In ra precision
và recall của lớp 0 rồi so với con số của lớp 1 trong tài liệu. Giải thích bằng lời vì sao hai
bộ số khác nhau, dù mô hình và dữ liệu không đổi chút nào.
'''
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia du lieu giong Bai 4
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)


# Huan luyen mo hinh
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Du doan tren tap kiem tra
y_pred = mo_hinh.predict(X_test)


# Danh gia voi lop duong la qua mon (lop 1)
precision_lop_1 = precision_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall_lop_1 = recall_score(
    y_test,
    y_pred,
    pos_label=1,
    zero_division=0
)


# Danh gia voi lop duong la rot mon (lop 0)
precision_lop_0 = precision_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)

recall_lop_0 = recall_score(
    y_test,
    y_pred,
    pos_label=0,
    zero_division=0
)


# In ket qua
print("SO SANH PRECISION VA RECALL GIUA HAI LOP")
print("=" * 50)

print("\nLop 1: Qua mon")
print(f"Precision = {precision_lop_1:.4f}")
print(f"Recall    = {recall_lop_1:.4f}")

print("\nLop 0: Rot mon")
print(f"Precision = {precision_lop_0:.4f}")
print(f"Recall    = {recall_lop_0:.4f}")


# Nhan xet
print("\nNHAN XET:")
print(
    "Hai bo chi so khac nhau vi Precision va Recall "
    "duoc tinh theo lop duong duoc chon."
)
print(
    "Khi doi lop duong tu 1 sang 0, vai tro cua TP, FP, FN "
    "cung thay doi theo lop dang duoc danh gia."
)
print(
    "Mo hinh va du lieu khong thay doi, nhung cach nhin nhan "
    "ket qua danh gia da thay doi."
)