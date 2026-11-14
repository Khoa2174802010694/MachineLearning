'''
Bài tập 3: Kernel đa thức
Trên bộ hai vầng trăng, thử SVC(kernel="poly", degree=d) với d bằng 2, 3 và 4, có chuẩn
hoá bằng make_pipeline. Với mỗi bậc, dò C trong 0.1, 1, 10, 100 bằng GridSearchCV.
Bậc nào tốt nhất? Có vượt được kernel RBF ở mục 5 không? Gợi ý, trong lưới tham số của
pipeline, tên tham số viết là "svc__C".
'''
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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

gia_tri_C = [0.1, 1, 10, 100]
ket_qua = []

for bac in [2, 3, 4]:
    pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel="poly", degree=bac)
    )

    luoi = {
        "svc__C": gia_tri_C
    }

    tim = GridSearchCV(
        pipeline,
        luoi,
        cv=5,
        scoring="accuracy"
    )

    tim.fit(X_tr, y_tr)

    accuracy_test = tim.score(X_te, y_te)

    ket_qua.append({
        "bac": bac,
        "C": tim.best_params_["svc__C"],
        "cv": tim.best_score_,
        "test": accuracy_test
    })

    print(f"Kernel da thuc bac {bac}")
    print("C tot nhat:", tim.best_params_["svc__C"])
    print(f"CV accuracy: {tim.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_test:.4f}")
    print()

ket_qua_df = pd.DataFrame(ket_qua)

tot_nhat = ket_qua_df.loc[ket_qua_df["cv"].idxmax()]

print("Bac tot nhat:", int(tot_nhat["bac"]))
print("C tot nhat:", tot_nhat["C"])
print(f"CV accuracy tot nhat: {tot_nhat['cv']:.4f}")
print(f"Test accuracy: {tot_nhat['test']:.4f}")

print()
print("So sanh voi SVM kernel RBF:")

rbf = SVC(kernel="rbf")
rbf.fit(X_tr, y_tr)

print(f"RBF test accuracy: {rbf.score(X_te, y_te):.4f}")
