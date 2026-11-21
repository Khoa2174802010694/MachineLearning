'''
Bài tập 3: Đổi biến đầu vào sang tuổi nhà
Dùng lại đúng công thức bình phương tối thiểu ở mục 5, nhưng đổi biến đầu vào từ
dien_tich sang tuoi_nha. Biến cần dự đoán vẫn là gia. In ra hệ số góc w và hệ số
chặn b với sáu chữ số sau dấu phẩy. Sau đó nhận xét dấu của w và giải thích bằng lời vì sao
nó mang dấu âm.
Cách nhanh nhất là chép tệp b3_cong_thuc.py sang thư mục baitap01 rồi đổi đúng một
dòng, dòng chọn cột cho biến x. Gợi ý cho phần giải thích, w là mức thay đổi của giá khi
tuổi nhà tăng thêm một năm. Khi chạy với dien_tich chúng ta được w = 0.078367 và
b = 0.401752, con số của sinh viên lần này sẽ khác hẳn nên đừng hoảng.
Có một câu hỏi thêm mà sinh viên nào làm xong sớm nên thử trả lời. Hãy vẽ biểu đồ phân
tán giữa tuoi_nha và gia rồi nhìn kỹ. Dấu âm của w là đúng, nhưng các điểm có thật sự
bám quanh một đường thẳng như ở biểu đồ diện tích không? Nếu không thì con số w này
đáng tin tới mức nào, và một hệ số có dấu đúng đã đủ để kết luận hai đại lượng có quan hệ
chặt chưa? Nộp tệp bai3.py, ảnh chụp kết quả và câu giải thích.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/gia_nha.csv")

# Chọn tuổi nhà làm biến đầu vào và giá nhà làm biến dự đoán
x = df["tuoi_nha"].to_numpy(dtype=float)
y = df["gia"].to_numpy(dtype=float)

# Tính hệ số góc và hệ số chặn bằng bình phương tối thiểu
x_tb = np.mean(x)
y_tb = np.mean(y)

w = np.sum((x - x_tb) * (y - y_tb)) / np.sum((x - x_tb) ** 2)
b = y_tb - w * x_tb

# Tính hệ số tương quan để đánh giá mức độ liên hệ
tuong_quan = np.corrcoef(x, y)[0, 1]

print("KET QUA HOI QUY TUYEN TINH")
print(f"He so goc w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print(f"He so tuong quan = {tuong_quan:.6f}")

if w < 0:
    print(
        "\nNhan xet: He so goc w mang dau am, "
        "cho thay gia nha co xu huong giam khi tuoi nha tang."
    )
elif w > 0:
    print(
        "\nNhan xet: He so goc w mang dau duong, "
        "cho thay gia nha co xu huong tang khi tuoi nha tang."
    )
else:
    print("\nNhan xet: He so goc w bang 0, khong co xu huong tuyen tinh.")

if abs(tuong_quan) < 0.3:
    print("Muc do tuong quan tuyen tinh giua tuoi nha va gia la yeu.")
elif abs(tuong_quan) < 0.7:
    print("Muc do tuong quan tuyen tinh giua tuoi nha va gia la trung binh.")
else:
    print("Muc do tuong quan tuyen tinh giua tuoi nha va gia la manh.")

# Vẽ biểu đồ phân tán và đường hồi quy
plt.figure(figsize=(8, 5))
plt.scatter(x, y, alpha=0.7, label="Du lieu thuc te")
plt.plot(
    x,
    w * x + b,
    color="red",
    label="Duong hoi quy"
)

plt.xlabel("Tuoi nha (nam)")
plt.ylabel("Gia nha (ty dong)")
plt.title("Moi quan he giua tuoi nha va gia nha")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("figures/bai3.png", dpi=150)
plt.show()

# Lưu kết quả
ket_qua = pd.DataFrame({
    "he_so_goc_w": [w],
    "he_so_chan_b": [b],
    "he_so_tuong_quan": [tuong_quan]
})

ket_qua.to_csv(
    "outputs/bai3_ket_qua.csv",
    index=False
)

with open("outputs/bai3_nhan_xet.txt", "w", encoding="utf-8") as f:
    f.write("KET QUA HOI QUY TUOI NHA VA GIA\n")
    f.write(f"He so goc w: {w:.6f}\n")
    f.write(f"He so chan b: {b:.6f}\n")
    f.write(f"He so tuong quan: {tuong_quan:.6f}\n")
    f.write(
        "Nhan xet: He so goc am cho thay gia nha co xu huong "
        "giam khi tuoi nha tang.\n"
    )
    f.write(
        "Muc do tuong quan tuyen tinh yeu, do do he so goc "
        "khong du de ket luan moi quan he chat che.\n"
    )
