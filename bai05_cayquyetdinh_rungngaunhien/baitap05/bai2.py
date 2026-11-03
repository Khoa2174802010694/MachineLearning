from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "khach_vay.csv"


def gini(y):
    if len(y) == 0:
        return 0.0

    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities ** 2)


def tim_cau_hoi(X, y):
    gini_goc = gini(y)
    giam_gini_tot_nhat = 0.0
    cot_tot_nhat = None
    nguong_tot_nhat = None

    for cot in X.columns:
        gia_tri = np.sort(X[cot].dropna().unique())

        if len(gia_tri) < 2:
            continue

        nguong_list = (gia_tri[:-1] + gia_tri[1:]) / 2

        for nguong in nguong_list:
            mask_trai = X[cot] <= nguong
            y_trai = y[mask_trai]
            y_phai = y[~mask_trai]

            if len(y_trai) == 0 or len(y_phai) == 0:
                continue

            gini_sau_chia = (
                len(y_trai) / len(y) * gini(y_trai)
                + len(y_phai) / len(y) * gini(y_phai)
            )
            giam_gini = gini_goc - gini_sau_chia

            if giam_gini > giam_gini_tot_nhat:
                giam_gini_tot_nhat = giam_gini
                cot_tot_nhat = cot
                nguong_tot_nhat = nguong

    return cot_tot_nhat, nguong_tot_nhat, giam_gini_tot_nhat


df = pd.read_csv(DATA_PATH)
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cot, nguong, muc_giam = tim_cau_hoi(X_train, y_train)

print("So mau train:", len(X_train))
print("Cau hoi tot nhat:", cot)
print("Nguong:", nguong)
print("Muc giam Gini:", muc_giam)
