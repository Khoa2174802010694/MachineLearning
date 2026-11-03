
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


df = pd.read_csv("data/khach_vay.csv")

X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

os.makedirs(
    "bai05_cayquyetdinh_rungngaunhien/baitap05",
    exist_ok=True
)

model_5_leaves = DecisionTreeClassifier(
    max_leaf_nodes=5,
    random_state=0
)

model_5_leaves.fit(X_train, y_train)

plt.figure(figsize=(18, 10))

plot_tree(
    model_5_leaves,
    feature_names=X.columns,
    class_names=[str(value) for value in model_5_leaves.classes_],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=9
)

plt.title("Decision Tree - Maximum 5 Leaves")

plt.savefig(
    "bai05_cayquyetdinh_rungngaunhien/baitap05/cay_nam_la.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

model_depth_2 = DecisionTreeClassifier(
    max_depth=2,
    random_state=0
)

model_depth_2.fit(X_train, y_train)

plt.figure(figsize=(16, 8))

plot_tree(
    model_depth_2,
    feature_names=X.columns,
    class_names=[str(value) for value in model_depth_2.classes_],
    filled=True,
    rounded=True,
    proportion=True,
    fontsize=10
)

plt.title("Decision Tree - Maximum Depth 2")

plt.savefig(
    "bai05_cayquyetdinh_rungngaunhien/baitap05/cay_do_sau_2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
