'''
Bài tập 4: Dò lưới mịn
Quanh cặp tốt nhất của mục 5, dò lại với C trong 3, 10, 30 và gamma trong 0.3, 1, 3. Điểm
kiểm định chéo tốt nhất có cao hơn 0.8711 không? Accuracy trên tập kiểm tra có đổi không?
Nhận xét về cái lợi của việc dò thật mịn.
'''
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


df = pd.read_csv("data/hai_vang_trang.csv")

X = df[["x1", "x2"]]
y = df["nhan"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

luoi = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}

tim = GridSearchCV(
    SVC(kernel="rbf"),
    luoi,
    cv=5,
    scoring="accuracy"
)

tim.fit(X_tr, y_tr)

bang = pd.DataFrame(tim.cv_results_["params"])
bang["diem"] = tim.cv_results_["mean_test_score"]

print("Diem kiem dinh cheo:")

print(
    bang.pivot(
        index="C",
        columns="gamma",
        values="diem"
    ).round(4).to_string()
)

print()

print("Cap tham so tot nhat:", tim.best_params_)
print(f"CV accuracy tot nhat: {tim.best_score_:.4f}")
print(f"Accuracy tren tap kiem tra: {tim.score(X_te, y_te):.4f}")
