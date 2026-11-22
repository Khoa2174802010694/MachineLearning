'''
Bài tập 2: Từ nào tố cáo mạnh nhất
Huấn luyện MultinomialNB trên toàn bộ 300 tin nhắn. Dùng thuộc tính
feature_log_prob_ để tính hiệu giữa hai lớp cho từng từ, rồi in ra 5 từ tố cáo tin
rác mạnh nhất và 5 từ cho thấy tin thường mạnh nhất.
Gợi ý, feature_log_prob_ là một mảng hai dòng, dòng 0 cho lớp 0 và dòng 1 cho lớp 1.
Lấy mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0] rồi sắp xếp.
'''
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Doc toan bo 300 tin nhan
df = pd.read_csv("data/tin_nhan.csv")

# Chuyen noi dung tin nhan thanh ma tran dem tu
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Huan luyen Multinomial Naive Bayes tren toan bo du lieu
mo_hinh = MultinomialNB()
mo_hinh.fit(X, y)

# Tinh chenh lech log xac suat cua tung tu giua hai lop
chenh_lech = (
    mo_hinh.feature_log_prob_[1]
    - mo_hinh.feature_log_prob_[0]
)

tu_vung = vectorizer.get_feature_names_out()

# Lay 5 tu co chenh lech lon nhat va nho nhat
chi_so_rac = chenh_lech.argsort()[-5:][::-1]
chi_so_thuong = chenh_lech.argsort()[:5]

print("5 TU TO CAO TIN RAC MANH NHAT")
print("=" * 42)
for i in chi_so_rac:
    print(f"{tu_vung[i]:<15} Chenh lech: {chenh_lech[i]:.4f}")

print("\n5 TU CHO THAY TIN THUONG MANH NHAT")
print("=" * 42)
for i in chi_so_thuong:
    print(f"{tu_vung[i]:<15} Chenh lech: {chenh_lech[i]:.4f}")