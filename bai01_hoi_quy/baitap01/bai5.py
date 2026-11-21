'''
Bài tập 5: Thử hai tốc độ học khác
Chép tệp b6_gradient_descent.py sang thư mục baitap01 rồi chạy lại hai lần. Lần thứ
nhất đặt tốc độ học bằng 0.001, lần thứ hai đặt bằng 1.02. Mỗi lần hãy ghi lại giá trị MSE
ở vòng lặp thứ 200, rồi giải thích vì sao hai con số đó khác nhau.
Trong tài liệu chúng ta chạy với tốc độ học 0.1 và sau 200 vòng thì MSE dừng ở 0.1790.
Hãy giữ nguyên so_vong = 200 và chỉ sửa đúng dòng gán toc_do_hoc, vì đổi hai thứ cùng
lúc thì không biết cái nào gây ra khác biệt. Gợi ý, hãy nhớ lại phần so sánh ba tốc độ học
ở mục 7, một bước quá ngắn thì đi chưa tới đáy còn một bước quá dài thì nhảy vọt qua đáy
rồi ra xa hơn. Với tốc độ học 1.02, con số MSE sẽ lớn tới mức kỳ quặc, và đó là kết quả
đúng chứ không phải máy hỏng.
Làm thêm nếu còn thời gian, hãy thử tốc độ học 0.5 rồi 0.9. Sinh viên sẽ thấy cả hai vẫn về
đúng 0.1790, nghĩa là ranh giới giữa chạy được và hỏng không nằm ở chỗ ta tưởng. Hãy dò
xem ranh giới đó nằm quanh con số nào. Nộp tệp bai5.py, hai ảnh chụp kết quả và phần
giải thích.
'''
import numpy as np
import pandas as pd

df = pd.read_csv("data/gia_nha.csv")

x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Chuan hoa x
x_tb = x_goc.mean()
x_std = x_goc.std()
x = (x_goc - x_tb) / x_std

n = len(x)

# Thu nghiem voi hai toc do hoc
for learning_rate in [0.001, 1.02]:
    print(f"\n===== Learning rate = {learning_rate} =====")

    w, b = 0.0, 0.0

    for i in range(200):
        y_pred = w * x + b
        error = y_pred - y

        grad_w = (2 / n) * (error * x).sum()
        grad_b = (2 / n) * error.sum()

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if i + 1 in [1, 2, 5, 10, 25, 50, 100, 200]:
            mse = ((w * x + b - y) ** 2).mean()
            print(
                f"Vong {i+1:3d}: "
                f"w={w:.4f}, b={b:.4f}, MSE={mse:.4f}"
            )

    # Dua w, b ve don vi ban dau
    w_goc = w / x_std
    b_goc = b - w * x_tb / x_std

    print(f"\nw = {w_goc:.6f}")
    print(f"b = {b_goc:.6f}")
    print(f"MSE cuoi cung = {mse:.6f}")
