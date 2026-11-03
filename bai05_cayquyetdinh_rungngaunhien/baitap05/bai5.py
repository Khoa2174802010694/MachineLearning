from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "khach_vay.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

rng = np.random.default_rng(0)
X_train_noise = X_train.copy()
X_test_noise = X_test.copy()

X_train_noise["so_ngau_nhien"] = rng.random(len(X_train_noise))
X_test_noise["so_ngau_nhien"] = rng.random(len(X_test_noise))

model = RandomForestClassifier(n_estimators=200, random_state=0)
model.fit(X_train_noise, y_train)

print("Test accuracy:", round(model.score(X_test_noise, y_test), 4))

gini_importance = pd.Series(
    model.feature_importances_,
    index=X_train_noise.columns,
    name="feature_importances"
).sort_values(ascending=False)

permutation = permutation_importance(
    model,
    X_test_noise,
    y_test,
    n_repeats=20,
    random_state=0,
    scoring="accuracy"
)

permutation_importance_mean = pd.Series(
    permutation.importances_mean,
    index=X_test_noise.columns,
    name="permutation_importance"
).sort_values(ascending=False)

print("\nFeature importance:")
print(gini_importance.to_string())

print("\nPermutation importance:")
print(permutation_importance_mean.to_string())

print("\nRank cua cot ngau nhien:")
print("Feature importance:", list(gini_importance.index).index("so_ngau_nhien") + 1)
print("Permutation importance:", list(permutation_importance_mean.index).index("so_ngau_nhien") + 1)
