'''
Bài tập 6: Lớp hiếm và xác suất tiên nghiệm
Từ bộ tin nhắn, tạo một bộ mới mất cân bằng nặng: giữ toàn bộ 180 tin thường nhưng chỉ
lấy 20 tin rác đầu tiên. Huấn luyện lại và chấm điểm.
In ra accuracy và recall của lớp rác. Nhận xét: accuracy có còn phản ánh đúng chất lượng
bộ lọc không? Gợi ý, dùng recall_score với pos_label=1, và nhớ lại cái bẫy accuracy ở
bài 2.
'''
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

# Doc du lieu
df = pd.read_csv("data/tin_nhan.csv")

# Tao bo du lieu mat can bang:
# Giữ toàn bộ 180 tin thường và chỉ lấy 20 tin rác đầu tiên.
tin_thuong = df[df["la_rac"] == 0]
tin_rac = df[df["la_rac"] == 1].head(20)
du_lieu = pd.concat([tin_thuong, tin_rac], ignore_index=True)

X = du_lieu["noi_dung"].astype(str)
y = du_lieu["la_rac"].astype(int)

# Chia train/test, giu ty le hai lop trong moi tap.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

# Vector hoa van ban va huan luyen MultinomialNB.
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Danh gia
accuracy = accuracy_score(y_test, y_pred)
recall_rac = recall_score(y_test, y_pred, pos_label=1)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print("KET QUA BAI 6")
print(f"So tin thuong: {len(tin_thuong)}")
print(f"So tin rac: {len(tin_rac)}")
print(f"So mau train: {len(X_train)}")
print(f"So mau test: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Recall lop rac: {recall_rac:.4f} ({recall_rac * 100:.2f}%)")
print(f"Tin thuong du doan dung: {tn}")
print(f"Tin rac du doan dung: {tp}")
print(f"Tin rac bi bo sot: {fn}")

nhan_xet = (
    "Nhan xet:\n"
    f"- Accuracy dat {accuracy * 100:.2f}%, nhung recall lop rac chi dat "
    f"{recall_rac * 100:.2f}%.\n"
    "- Accuracy co the tao cam giac mo hinh hoat dong tot khi lop thuong chiem "
    "da so, nhung no khong cho thay day du kha nang phat hien lop rac.\n"
    f"- Trong tap test, mo hinh bo sot {fn}/{fn + tp} tin rac. "
    "Vi vay can xem them recall cua lop rac, khong nen chi dua vao accuracy.\n"
)

Path("outputs").mkdir(exist_ok=True)
pd.DataFrame([{
    "so_tin_thuong": len(tin_thuong),
    "so_tin_rac": len(tin_rac),
    "so_train": len(X_train),
    "so_test": len(X_test),
    "accuracy": accuracy,
    "recall_lop_rac": recall_rac,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp
}]).to_csv("outputs/bai6_ket_qua.csv", index=False, encoding="utf-8-sig")

with open("outputs/bai6_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write(nhan_xet)
