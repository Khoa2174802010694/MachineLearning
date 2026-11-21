'''
Bài tập 2: Vẽ hàm sigmoid
Vẽ đồ thị hàm sigmoid với z chạy từ −8 tới 8, lưu thành tệp sigmoid.png. Trên cùng hình,
vẽ thêm một đường ngang tại mức 0.5 và một đường dọc tại z = 0.
Gợi ý, dùng np.linspace, plt.plot, plt.axhline, plt.axvline và plt.savefig
'''
import numpy as np
import matplotlib.pyplot as plt
import os

# Ham sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Tao du lieu z tu -8 den 8
z = np.linspace(-8, 8, 200)
y = sigmoid(z)

# Ve do thi
plt.figure(figsize=(8, 5))
plt.plot(z, y, label="Sigmoid", linewidth=2)

# Them duong tham chieu
plt.axhline(y=0.5, color="red", linestyle="--", label="y = 0.5")
plt.axvline(x=0, color="green", linestyle="--", label="z = 0")

# Tieu de va nhan truc
plt.title("Ham Sigmoid")
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.legend()

# Luu hinh vao thu muc figures
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/sigmoid.png", dpi=300, bbox_inches="tight")

plt.show()
