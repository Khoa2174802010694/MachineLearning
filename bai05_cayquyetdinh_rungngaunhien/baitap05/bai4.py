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

gia_tri_min_leaf = [1, 2, 5, 10, 20, 40]
ket_qua = []

for min_leaf in gia_tri_min_leaf:
    model = DecisionTreeClassifier(
        min_samples_leaf=min_leaf,
        random_state=0
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    ket_qua.append((min_leaf, scores.mean(), scores.std()))

for min_leaf, diem_tb, do_lech in ket_qua:
    print(
        f"min_samples_leaf={min_leaf:2d}: "
        f"CV accuracy = {diem_tb:.4f} (+/- {do_lech:.4f})"
    )

min_leaf_tot_nhat, diem_cv_tot_nhat, _ = max(
    ket_qua, key=lambda item: item[1]
)

model_pruned = DecisionTreeClassifier(
    min_samples_leaf=min_leaf_tot_nhat,
    random_state=0
)
model_pruned.fit(X_train, y_train)

model_depth_5 = DecisionTreeClassifier(max_depth=5, random_state=0)
model_depth_5.fit(X_train, y_train)

print("\nMin samples leaf tot nhat:", min_leaf_tot_nhat)
print("CV accuracy tot nhat:", round(diem_cv_tot_nhat, 4))
print("Test accuracy - min_samples_leaf:", round(model_pruned.score(X_test, y_test), 4))
print("Test accuracy - max_depth=5:", round(model_depth_5.score(X_test, y_test), 4))
