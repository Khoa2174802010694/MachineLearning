'''
Bài tập 4: Tự tính bốn thước đo
Chép tệp c4_danh_gia.py sang baitap02 rồi bỏ hết các hàm accuracy_score,
precision_score, recall_score, f1_score của scikit-learn đi. Tự tính bốn con số đó
chỉ từ bốn số TP, TN, FP, FN bằng công thức ở mục 5.
So kết quả tự tính với kết quả thư viện in ra trong tài liệu, hai bên phải khớp. Nộp cả phần
mã tự tính và ảnh chụp kết quả.
'''
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Doc du lieu
df = pd.read_csv("data/sinh_vien.csv")

X = df[["gio_on"]]
y = df["qua_mon"]


# Chia du lieu thanh tap huan luyen va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y,
)

print("So sinh vien de hoc:", len(X_train))
print("So sinh vien de kiem tra:", len(X_test))
print()


# Khoi tao va huan luyen mo hinh
mo_hinh = LogisticRegression()
mo_hinh.fit(X_train, y_train)


# Du doan ket qua tren tap kiem tra
y_pred = mo_hinh.predict(X_test)


# Lay bon gia tri tu ma tran nham lan
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("Ma tran nham lan:")
print(f" TN = {tn:2d} | Doan rot, that su rot")
print(f" FP = {fp:2d} | Doan qua, that ra rot")
print(f" FN = {fn:2d} | Doan rot, that ra qua")
print(f" TP = {tp:2d} | Doan qua, that su qua")
print()


# Tu tinh bon thuoc do bang cong thuc
n = len(y_test)

accuracy = (tp + tn) / n

precision = tp / (tp + fp) if (tp + fp) > 0 else 0

recall = tp / (tp + fn) if (tp + fn) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# In ket qua
print("KET QUA TU TINH BANG CONG THUC:")
print(f"Accuracy  = {accuracy:.4f}")
print(f"Precision = {precision:.4f}")
print(f"Recall    = {recall:.4f}")
print(f"F1        = {f1:.4f}")
print()


# Hien thi phep tinh chi tiet
print("CHI TIET CAC PHEP TINH:")

print(
    f"Accuracy  = ({tp} + {tn}) / {n} "
    f"= {accuracy:.4f}"
)

print(
    f"Precision = {tp} / ({tp} + {fp}) "
    f"= {precision:.4f}"
)

print(
    f"Recall    = {tp} / ({tp} + {fn}) "
    f"= {recall:.4f}"
)

print(
    f"F1        = 2 * {precision:.4f} * {recall:.4f} "
    f"/ ({precision:.4f} + {recall:.4f}) "
    f"= {f1:.4f}"
)