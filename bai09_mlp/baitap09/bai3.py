'''
Bài tập 3: So các hàm kích hoạt
Trên bộ hai vầng trăng, dùng mạng hai lớp ẩn 100 nơ-ron với alpha=1.0 như mục 4. Lần
lượt đặt activation bằng "logistic", "tanh" và "relu". Ghi lại accuracy trên tập học
và tập kiểm tra. Hàm nào cho kết quả tốt nhất ở đây?
'''
import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

ket_qua = []

print("SO SANH CAC HAM KICH HOAT")
print("Activation       Tap hoc    Tap kiem tra")

# Thử nghiệm ba hàm kích hoạt
for activation in ["logistic", "tanh", "relu"]:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation=activation,
            alpha=1.0,
            solver="lbfgs",
            max_iter=20000,
            random_state=0
        )
    )

    mang.fit(X_tr, y_tr)

    acc_hoc = accuracy_score(y_tr, mang.predict(X_tr))
    acc_kiem_tra = accuracy_score(y_te, mang.predict(X_te))

    ket_qua.append({
        "activation": activation,
        "accuracy_tap_hoc": acc_hoc,
        "accuracy_tap_kiem_tra": acc_kiem_tra
    })

    print(
        f"{activation:15s} "
        f"{acc_hoc:.4f}     "
        f"{acc_kiem_tra:.4f}"
    )

# Lưu kết quả
ket_qua = pd.DataFrame(ket_qua)
ket_qua.to_csv(
    "outputs/bai3_so_sanh_activation.csv",
    index=False
)

# Xác định hàm kích hoạt tốt nhất trên tập kiểm tra
tot_nhat = ket_qua.loc[
    ket_qua["accuracy_tap_kiem_tra"].idxmax()
]

print()
print("HAM KICH HOAT TOT NHAT")
print("Activation:", tot_nhat["activation"])
print(
    "Accuracy tap kiem tra:",
    f"{tot_nhat['accuracy_tap_kiem_tra']:.4f}"
)
