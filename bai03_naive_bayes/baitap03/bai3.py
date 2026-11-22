'''
Bài tập 3: Đổi tham số làm trơn
Chạy lại bộ lọc với alpha lần lượt bằng 0.001, 0.1, 1.0, 10.0 và 100.0. Với mỗi giá trị, in
ra accuracy, precision và recall trên tập kiểm tra.
Nhận xét một câu: alpha lớn quá thì chuyện gì xảy ra, và vì sao? Gợi ý cho phần giải thích,
alpha rất lớn làm mọi xác suất từ xích lại gần nhau.
'''
import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Doc du lieu
df = pd.read_csv("data/tin_nhan.csv")

X = df["noi_dung"]
y = df["la_rac"]


# Chia du lieu thanh tap huan luyen va tap kiem tra
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)


# Chuyen van ban thanh ma tran dac trung
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Cac gia tri alpha can thu nghiem
cac_alpha = [0.001, 0.1, 1.0, 10.0, 100.0]

ket_qua = []

print("SO SANH CAC GIA TRI ALPHA")
print("=" * 55)
print(f"{'Alpha':<12}{'Accuracy':<15}{'Precision':<15}{'Recall':<15}")
print("-" * 55)


# Huan luyen va danh gia voi tung alpha
for alpha in cac_alpha:
    mo_hinh = MultinomialNB(alpha=alpha)
    mo_hinh.fit(X_train_vec, y_train)

    y_pred = mo_hinh.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, pos_label=1, zero_division=0
    )

    ket_qua.append({
        "alpha": alpha,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    })

    print(
        f"{alpha:<12.3f}"
        f"{accuracy:<15.4f}"
        f"{precision:<15.4f}"
        f"{recall:<15.4f}"
    )


# Nhan xet ve tac dong cua alpha
nhan_xet = (
    "Khi alpha qua lon, cac xac suat tu co xu huong xich lai gan nhau. "
    "Mo hinh mat dan kha nang phan biet tu dac trung giua tin rac va "
    "tin thuong, do do hieu qua phan loai co the giam."
)

print("\nNHAN XET:")
print(nhan_xet)
os.makedirs("outputs", exist_ok=True)

df_ket_qua = pd.DataFrame(ket_qua)
df_ket_qua.to_csv(
    "outputs/bai3_ket_qua.csv",
    index=False,
    encoding="utf-8-sig"
)

with open("outputs/bai3_nhan_xet.txt", "w", encoding="utf-8") as file:
    file.write("BAI TAP 3: THAY DOI THAM SO LAM TRON ALPHA\n\n")
    file.write(df_ket_qua.to_string(index=False))
    file.write("\n\nNHAN XET:\n")
    file.write(nhan_xet)
