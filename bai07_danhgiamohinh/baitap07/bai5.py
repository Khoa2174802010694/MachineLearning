'''
Bài tập 5: Cân bằng lớp cho mô hình khác
Thêm class_weight="balanced" cho RandomForestClassifier và cho SVC. 
So precision, recall, F1 trên tập kiểm tra trước và sau khi cân bằng. Mô hình nào thay đổi nhiều
nhất?
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    "Random Forest": [
        RandomForestClassifier(
            n_estimators=200,
            random_state=0
        ),
        RandomForestClassifier(
            n_estimators=200,
            random_state=0,
            class_weight="balanced"
        )
    ],
    "SVM RBF": [
        make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        ),
        make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced")
        )
    ]
}

ket_qua = {}

print("So sanh truoc va sau khi can bang lop:")
print("Mo hinh          Precision  Recall     F1")

for ten, cap_mh in mo_hinh.items():
    ket_qua[ten] = []

    for i, mh in enumerate(cap_mh):
        du = mh.fit(X_tr, y_tr).predict(X_te)

        p = precision_score(y_te, du, zero_division=0)
        r = recall_score(y_te, du, zero_division=0)
        f = f1_score(y_te, du, zero_division=0)

        ket_qua[ten].append((p, r, f))

        trang_thai = "Can bang" if i == 1 else "Binh thuong"

        print(
            f"{ten + ' ' + trang_thai:24s}"
            f"{p:.4f}     {r:.4f}     {f:.4f}"
        )

print("\nMuc thay doi sau khi can bang:")

for ten, kq in ket_qua.items():
    truoc = kq[0]
    sau = kq[1]

    thay_doi = [
        abs(sau[i] - truoc[i])
        for i in range(3)
    ]

    print(
        f"{ten}: "
        f"Precision {sau[0] - truoc[0]:+.4f}, "
        f"Recall {sau[1] - truoc[1]:+.4f}, "
        f"F1 {sau[2] - truoc[2]:+.4f}"
    )

    print(f"Do thay doi lon nhat: {max(thay_doi):.4f}")
