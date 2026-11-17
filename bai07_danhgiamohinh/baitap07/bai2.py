'''
Bài tập 2: Chữ số viết tay
Lấy lại SVM nhận dạng chữ số của bài 6, in classification_report cho cả 10 chữ số.
Chữ số nào có recall thấp nhất? Chữ số nào có precision thấp nhất? Vì sao với bộ dữ liệu
này, macro avg và weighted avg gần như bằng nhau?
'''
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

svm = SVC(kernel="rbf")
svm.fit(X_tr, y_tr)

du = svm.predict(X_te)

print("Classification report:")
print(
    classification_report(
        y_te,
        du,
        labels=list(range(10)),
        digits=4
    )
)

bao_cao = classification_report(
    y_te,
    du,
    labels=list(range(10)),
    output_dict=True,
    zero_division=0
)

recall = {
    int(k): v["recall"]
    for k, v in bao_cao.items()
    if k.isdigit()
}

precision = {
    int(k): v["precision"]
    for k, v in bao_cao.items()
    if k.isdigit()
}

print("Chu so co recall thap nhat:", min(recall, key=recall.get))
print("Recall thap nhat:", min(recall.values()))
print("Chu so co precision thap nhat:", min(precision, key=precision.get))
print("Precision thap nhat:", min(precision.values()))
