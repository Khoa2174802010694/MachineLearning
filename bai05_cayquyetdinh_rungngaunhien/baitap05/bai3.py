from pathlib import Path

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "khach_vay.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

ket_qua = []

for do_sau in range(1, 16):
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=do_sau,
        random_state=0
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    ket_qua.append((do_sau, scores.mean(), scores.std()))

for do_sau, diem_tb, do_lech in ket_qua:
    print(
        f"Do sau {do_sau:2d}: "
        f"CV accuracy = {diem_tb:.4f} (+/- {do_lech:.4f})"
    )

do_sau_tot_nhat, diem_cv_tot_nhat, _ = max(ket_qua, key=lambda item: item[1])

model_tot_nhat = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=do_sau_tot_nhat,
    random_state=0
)
model_tot_nhat.fit(X_train, y_train)

print("\nDo sau tot nhat:", do_sau_tot_nhat)
print("CV accuracy tot nhat:", round(diem_cv_tot_nhat, 4))
print("Test accuracy:", round(model_tot_nhat.score(X_test, y_test), 4))
