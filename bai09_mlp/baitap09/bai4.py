'''
Bài tập 4: Tốc độ học
Trên bộ chữ số, dùng mạng một lớp ẩn 100 nơ-ron, lần lượt đặt learning_rate_init bằng
0.0001, 0.001, 0.01 và 0.1. Với mỗi giá trị, ghi số vòng học n_iter_, mất mát cuối cùng
và accuracy trên tập kiểm tra. Vẽ bốn đường loss_curve_ lên cùng một hình và lưu thành
toc_do_hoc.png. Tốc độ học quá lớn và quá nhỏ gây ra chuyện gì?
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
os.makedirs("figures", exist_ok=True)

X, y = load_digits(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

toc_do_hoc = [0.0001, 0.001, 0.01, 0.1]
ket_qua = []
duong_loss = {}

print("SO SANH CAC TOC DO HOC")
print("Learning rate   So vong hoc   Loss cuoi   Accuracy")

# Huấn luyện mạng với từng tốc độ học
for lr in toc_do_hoc:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(100,),
            learning_rate_init=lr,
            solver="adam",
            max_iter=1000,
            random_state=0
        )
    )

    mang.fit(X_tr, y_tr)

    mlp = mang[-1]
    du_doan = mang.predict(X_te)

    so_vong = mlp.n_iter_
    loss_cuoi = mlp.loss_curve_[-1]
    accuracy = accuracy_score(y_te, du_doan)

    duong_loss[lr] = mlp.loss_curve_

    ket_qua.append({
        "learning_rate": lr,
        "so_vong_hoc": so_vong,
        "loss_cuoi": loss_cuoi,
        "accuracy_tap_kiem_tra": accuracy
    })

    print(
        f"{lr:<15g} "
        f"{so_vong:<13d} "
        f"{loss_cuoi:<11.6f} "
        f"{accuracy:.4f}"
    )

# Lưu bảng kết quả
ket_qua = pd.DataFrame(ket_qua)
ket_qua.to_csv(
    "outputs/bai4_ket_qua.csv",
    index=False
)

# Vẽ bốn đường Loss trên cùng một biểu đồ
plt.figure(figsize=(10, 6))

for lr, loss in duong_loss.items():
    plt.plot(
        range(1, len(loss) + 1),
        loss,
        label=f"Learning rate = {lr}"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("So sanh Loss voi cac toc do hoc")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/toc_do_hoc.png", dpi=150)
plt.close()

# Xác định tốc độ học có Accuracy cao nhất
tot_nhat = ket_qua.loc[
    ket_qua["accuracy_tap_kiem_tra"].idxmax()
]

print()
print("TOC DO HOC CO ACCURACY CAO NHAT")
print("Learning rate:", tot_nhat["learning_rate"])
print(
    "Accuracy:",
    f"{tot_nhat['accuracy_tap_kiem_tra']:.4f}"
)

