'''
Bài tập 3: Chọn ngưỡng theo yêu cầu
Nhà trường yêu cầu tìm ra ít nhất 80% số bạn bỏ học. Dùng xác suất từ cross_val_predict
trên tập học như mục 5, tìm ngưỡng lớn nhất cho recall từ 0.8 trở lên.
Áp ngưỡng đó cho tập kiểm tra một lần. Recall có đạt 0.8 không? Precision và số bạn bị
báo bằng bao nhiêu?
'''
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/sinh_vien_bo_hoc.csv")

X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=17,
    stratify=y
)

lr = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

xs_hoc = cross_val_predict(
    lr,
    X_tr,
    y_tr,
    cv=5,
    method="predict_proba"
)[:, 1]

cac_nguong = np.sort(np.unique(xs_hoc))[::-1]

nguong_hop_le = []

for t in cac_nguong:
    du_hoc = (xs_hoc >= t).astype(int)
    rec = recall_score(y_tr, du_hoc, zero_division=0)

    if rec >= 0.8:
        nguong_hop_le.append(t)

t_tot = max(nguong_hop_le)

du_hoc = (xs_hoc >= t_tot).astype(int)

print("Ket qua chon nguong tren tap hoc:")
print(f"Nguong lon nhat: {t_tot:.6f}")
print(f"Recall kiem dinh cheo: {recall_score(y_tr, du_hoc):.4f}")

lr.fit(X_tr, y_tr)
xs_te = lr.predict_proba(X_te)[:, 1]
du_te = (xs_te >= t_tot).astype(int)

tp = int(((du_te == 1) & (y_te.to_numpy() == 1)).sum())
fp = int(((du_te == 1) & (y_te.to_numpy() == 0)).sum())
fn = int(((du_te == 0) & (y_te.to_numpy() == 1)).sum())

print("\nKet qua tren tap kiem tra:")
print(f"Nguong: {t_tot:.6f}")
print(f"So ban bi bao: {du_te.sum()}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
print(f"Precision: {precision_score(y_te, du_te, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_te, du_te, zero_division=0):.4f}")
print(f"F1: {f1_score(y_te, du_te, zero_division=0):.4f}")

'''
Ngưỡng được chọn trên tập huấn luyện bằng dự đoán kiểm định chéo, không dựa vào nhãn của tập kiểm tra. 
Vì vậy, Recall trên tập kiểm tra có thể thấp hoặc cao hơn 0,8. 
Nếu không đạt 0,8, điều đó không có nghĩa là cách chọn ngưỡng sai, mà cho thấy khả năng tổng quát hóa của mô hình trên dữ liệu chưa thấy còn hạn chế.
'''
