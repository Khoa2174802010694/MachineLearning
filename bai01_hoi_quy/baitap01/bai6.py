'''
Bài tập 6: Viết hàm dự đoán có cảnh báo
Viết một hàm tên du_doan_gia(dien_tich) nhận vào diện tích và trả về giá dự đoán theo
mô hình một biến. Bên trong hàm, nếu diện tích nằm ngoài khoảng từ 35.5 tới 117.5 mét
vuông thì in thêm một dòng cảnh báo trước khi trả về kết quả. Gọi thử hàm với ba giá trị là
60, 80 và 200 mét vuông rồi in cả ba kết quả ra màn hình.
Sinh viên không phải huấn luyện lại gì cả, chỉ cần dùng lại hai con số đã có là w = 0.078367
và b = 0.401752. Gợi ý, hãy khai báo hàm bằng def, trả kết quả bằng return, và viết cảnh
báo trong một câu lệnh if với điều kiện dien_tich < 35.5 or dien_tich > 117.5. Có
một mốc để sinh viên tự kiểm tra, gọi hàm với 80 mét vuông phải ra 6.671 tỷ đồng, đúng
bằng con số ở mục 5. Lý do phải cảnh báo là căn 200 mét vuông nằm xa mọi căn trong dữ
liệu, mô hình chưa từng thấy vùng đó nên con số nó đưa ra chỉ là suy đoán kéo dài. Nộp tệp
bai6.py và ảnh chụp ba kết quả.
'''
def du_doan_gia(dien_tich):
    w = 0.078367
    b = 0.401752

    if dien_tich < 35.5 or dien_tich > 117.5:
        print(
            f"Canh bao: Dien tich {dien_tich} m2 nam ngoai "
            "pham vi du lieu huan luyen (35.5 - 117.5 m2)."
        )

    gia_du_doan = w * dien_tich + b
    return gia_du_doan


if __name__ == "__main__":
    for dien_tich in [60, 80, 200]:
        gia = du_doan_gia(dien_tich)
        print(f"Gia du doan cho can {dien_tich} m2: {gia:.3f} ty dong")
