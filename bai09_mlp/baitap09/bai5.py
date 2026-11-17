'''
Bài tập 5: Dò lưới cho mạng nơ‐ron
Dùng GridSearchCV trên bộ chữ số với hidden_layer_sizes trong (10,),
(50,), (100,), (50, 50) và alpha trong 0.0001, 0.01, 1. Mô hình là
make_pipeline(StandardScaler(), MLPClassifier(...)), nên tên tham số trong
lưới viết là "mlpclassifier__alpha".
Cặp nào tốt nhất? Accuracy trên tập kiểm tra có vượt được SVM không?
'''
import os
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

os.makedirs("outputs", exist_ok=True)

X, y = load_digits(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Xây dựng pipeline cho mạng nơ-ron
mang = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        max_iter=1000,
        random_state=0
    )
)

# Các cấu hình cần thử nghiệm
luoi_tham_so = {
    "mlpclassifier__hidden_layer_sizes": [
        (10,),
        (50,),
        (100,),
        (50, 50)
    ],
    "mlpclassifier__alpha": [
        0.0001,
        0.01,
        1
    ]
}

# Dò lưới bằng kiểm định chéo 5 phần
grid = GridSearchCV(
    mang,
    luoi_tham_so,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    refit=True
)

grid.fit(X_tr, y_tr)

# Lấy mô hình MLP tốt nhất
mlp_tot_nhat = grid.best_estimator_
acc_mlp = accuracy_score(
    y_te,
    mlp_tot_nhat.predict(X_te)
)

print("KET QUA GRID SEARCH")
print("Tham so tot nhat:", grid.best_params_)
print("Accuracy kiem dinh cheo:", f"{grid.best_score_:.4f}")
print("Accuracy MLP tren tap kiem tra:", f"{acc_mlp:.4f}")

# Lưu kết quả thử nghiệm các cấu hình
bang_grid = pd.DataFrame(grid.cv_results_)

bang_grid = bang_grid[[
    "param_mlpclassifier__hidden_layer_sizes",
    "param_mlpclassifier__alpha",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]]

bang_grid = bang_grid.rename(columns={
    "param_mlpclassifier__hidden_layer_sizes": "hidden_layer_sizes",
    "param_mlpclassifier__alpha": "alpha",
    "mean_test_score": "accuracy_trung_binh",
    "std_test_score": "do_lech_chuan",
    "rank_test_score": "thu_hang"
})

bang_grid.to_csv(
    "outputs/bai5_grid_search.csv",
    index=False
)

# Huấn luyện SVM trên cùng tập dữ liệu
svm = make_pipeline(
    StandardScaler(),
    SVC()
)

svm.fit(X_tr, y_tr)

acc_svm = accuracy_score(
    y_te,
    svm.predict(X_te)
)

# So sánh MLP tốt nhất với SVM
print()
print("SO SANH MLP VA SVM")
print(f"MLP: {acc_mlp:.4f}")
print(f"SVM: {acc_svm:.4f}")

if acc_mlp > acc_svm:
    print("MLP co accuracy cao hon SVM.")
elif acc_mlp < acc_svm:
    print("SVM co accuracy cao hon MLP.")
else:
    print("MLP va SVM co accuracy bang nhau.")

# Lưu kết quả so sánh
so_sanh = pd.DataFrame([
    {
        "mo_hinh": "MLP",
        "accuracy": acc_mlp
    },
    {
        "mo_hinh": "SVM",
        "accuracy": acc_svm
    }
])

so_sanh.to_csv(
    "outputs/bai5_so_sanh_mlp_svm.csv",
    index=False
)

# Lưu cấu hình tốt nhất và nhận xét kết quả
with open("outputs/bai5_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write("KET QUA GRID SEARCH\n")
    f.write(f"Tham so tot nhat: {grid.best_params_}\n")
    f.write(f"Accuracy kiem dinh cheo: {grid.best_score_:.4f}\n")
    f.write(f"Accuracy MLP tren tap kiem tra: {acc_mlp:.4f}\n")
    f.write(f"Accuracy SVM tren tap kiem tra: {acc_svm:.4f}\n")

    if acc_mlp > acc_svm:
        f.write("MLP co accuracy cao hon SVM.\n")
    elif acc_mlp < acc_svm:
        f.write("SVM co accuracy cao hon MLP.\n")
    else:
        f.write("MLP va SVM co accuracy bang nhau.\n")


