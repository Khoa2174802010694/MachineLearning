'''
Bài tập 4: Tự vẽ đường ROC
Dùng roc_curve lấy hai mảng fpr và tpr của hồi quy logistic và của rừng ngẫu nhiên. Vẽ
cả hai đường cùng đường chéo lên một hình bằng plt.plot, ghi AUC vào chú thích, lưu
thành roc.png. Đường nào ôm sát góc trên bên trái hơn? Có đoạn nào hai đường cắt nhau
không?
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression()
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=0
    )
}

plt.figure(figsize=(8, 6))

for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    xs = mh.predict_proba(X_te)[:, 1]

    fpr, tpr, nguong = roc_curve(y_te, xs)
    auc = roc_auc_score(y_te, xs)

    plt.plot(
        fpr,
        tpr,
        label=f"{ten} (AUC = {auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Duong ngau nhien"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc.png", dpi=300)
plt.show()
print("Da luu bieu do: roc.png")
