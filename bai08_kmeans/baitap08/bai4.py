'''
Bài tập 4: Lưu ảnh đã nén
Dựa vào k5_nen_anh.py, tạo ảnh 16 màu rồi lưu bằng plt.imsave("anh_16_mau.png",
anh_moi), với anh_moi là mảng có dạng (427, 640, 3). Lưu cả ảnh gốc, rồi so dung
lượng hai tệp.
Làm lại với một bức ảnh của chính em, đọc bằng plt.imread. Chú ý ảnh PNG có thể có
kênh thứ tư là độ trong suốt, phải bỏ kênh đó trước khi phân cụm.
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
os.makedirs("figures", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

def chuan_hoa_anh(anh):
    anh = np.asarray(anh)

    if anh.ndim == 2:
        anh = np.stack([anh, anh, anh], axis=-1)

    if anh.shape[2] == 4:
        anh = anh[:, :, :3]

    anh = anh[:, :, :3]

    if np.issubdtype(anh.dtype, np.integer):
        anh = anh.astype(float) / np.iinfo(anh.dtype).max
    else:
        anh = np.clip(anh.astype(float), 0, 1)

    return anh

def nen_anh(anh, ten, ten_anh_goc, ten_anh_nen):
    anh = chuan_hoa_anh(anh)

    cao, rong, _ = anh.shape
    diem = anh.reshape(-1, 3)

    plt.imsave(ten_anh_goc, anh)

    rng = np.random.default_rng(0)

    mau = diem[
        rng.choice(
            len(diem),
            min(5000, len(diem)),
            replace=False
        )
    ]

    km = KMeans(
        n_clusters=16,
        n_init=4,
        random_state=0
    ).fit(mau)

    nhan = km.predict(diem)

    anh_moi = km.cluster_centers_[nhan].reshape(
        cao,
        rong,
        3
    )

    plt.imsave(ten_anh_nen, anh_moi)

    size_goc = os.path.getsize(ten_anh_goc)
    size_nen = os.path.getsize(ten_anh_nen)

    print(f"\n{ten}")
    print(f"Kich thuoc anh: {cao} x {rong}")
    print(f"Anh goc: {size_goc:,} bytes")
    print(f"Anh nen: {size_nen:,} bytes")
    print(f"Chenh lech: {size_goc - size_nen:,} bytes")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(anh)
    plt.title("Anh goc")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(anh_moi)
    plt.title("Anh nen 16 mau")
    plt.axis("off")

    plt.tight_layout()

    ten_hinh = ten_anh_nen.replace(
        ".png",
        "_so_sanh.png"
    )

    plt.savefig(ten_hinh, dpi=200)
    plt.show()

    return {
        "anh": ten,
        "chieu_cao": cao,
        "chieu_rong": rong,
        "dung_luong_goc_bytes": size_goc,
        "dung_luong_nen_bytes": size_nen
    }

ket_qua = []

anh = load_sample_image("china.jpg")

ket_qua.append(
    nen_anh(
        anh,
        "china.jpg",
        "figures/bai4_anh_goc.png",
        "figures/bai4_anh_16_mau.png"
    )
)

duong_dan = "data/anh_cua_em.png"

if os.path.exists(duong_dan):
    anh_rieng = plt.imread(duong_dan)

    ket_qua.append(
        nen_anh(
            anh_rieng,
            "anh_cua_em.png",
            "figures/bai4_anh_rieng_goc.png",
            "figures/bai4_anh_rieng_16_mau.png"
        )
    )
else:
    print()
    print(f"Chua tim thay anh rieng: {duong_dan}")
    print("Hay dat anh cua ban vao thu muc data.")

bang = pd.DataFrame(ket_qua)

bang.to_csv(
    "outputs/bai4_dung_luong_anh.csv",
    index=False
)

print("\nBANG SO SANH DUNG LUONG ANH")
print(bang.to_string(index=False))
