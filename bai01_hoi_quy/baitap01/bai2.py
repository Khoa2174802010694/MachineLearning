'''
Bài tập 2: Vẽ biểu đồ phân tán theo số phòng
Vẽ biểu đồ phân tán với trục hoành là cột so_phong và trục tung là cột gia. Đặt tên cho
hai trục và đặt tiêu đề cho hình, vì một hình không có nhãn thì người xem không hiểu. Lưu
hình thành tệp bai2.png rồi viết một câu nhận xét về xu hướng sinh viên nhìn thấy.
Bài này dùng matplotlib.pyplot giống hệt cách chúng ta đã vẽ hình đầu tiên của bài thực
hành, chỉ đổi cột đưa vào scatter. Gợi ý, hãy gọi plt.savefig("bai2.png", dpi=150)
trước khi gọi plt.show(), vì lệnh show đóng hình lại và sau đó lưu ra sẽ được một tệp trắng.
Câu nhận xét viết bằng tiếng Việt, đặt trong lệnh print để nó hiện ra trong ảnh chụp. Nộp
tệp bai2.py, tệp hình bai2.png và ảnh chụp kết quả.
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/gia_nha.csv")

# Vẽ biểu đồ phân tán số phòng và giá nhà
plt.figure(figsize=(8, 5))
plt.scatter(
    df["so_phong"],
    df["gia"],
    alpha=0.7
)

plt.xlabel("So phong")
plt.ylabel("Gia nha (ty dong)")
plt.title("Moi quan he giua so phong va gia nha")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Lưu biểu đồ trước khi hiển thị
plt.savefig("figures/bai2.png", dpi=150)
plt.show()

# Nhận xét xu hướng dữ liệu
tuong_quan = df["so_phong"].corr(df["gia"])

print("KET QUA PHAN TICH")
print(f"He so tuong quan giua so phong va gia nha: {tuong_quan:.4f}")
print(
    "Nhan xet: Bieu do cho thay xu huong tuong quan thuan, "
    "so phong cang tang thi gia nha nhin chung cang cao."
)
