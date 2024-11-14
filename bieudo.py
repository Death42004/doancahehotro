import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Đọc tệp CSV
df = pd.read_csv("DATA.csv")

# Xóa các cột không cần thiết (điều chỉnh dựa trên các cột hiện có)
df.drop(['Khu vuc ma hoa', 'duong ma hoa'], axis=1, inplace=True)

# Loại bỏ khoảng trắng ở đầu và cuối giá trị của cột
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# Loại bỏ các hàng có giá trị bị thiếu
df.dropna(inplace=True)

# Đặt lại chỉ số
df.reset_index(drop=True, inplace=True)

# Đổi tên các cột
df_rename = {
    'Loai': 'Loai',
    'Gia(USD)': 'Gia(USD)',
    'So phong ngu': 'So phong ngu',
    'So phong tam': 'So phong tam',
    'Dien tich(feet)': 'Dien tich(feet)',
    'Khu vuc': 'Khu vuc',
    'Duong': 'Duong',
    'Vi do': 'Vi do',
    'Kinh do': 'Kinh do'
}
df = df.rename(columns=df_rename)

# Chia giá trị của cột 'Gia(USD)' cho 1000
df['Gia(USD)'] = df['Gia(USD)'] / 1000

# Xử lý ngoại lệ (outliers) bằng quy tắc IQR
Q1 = df['Gia(USD)'].quantile(0.25)
Q3 = df['Gia(USD)'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.85 * IQR
upper_bound = Q3 + 1.85 * IQR
df = df[(df['Gia(USD)'] >= lower_bound) & (df['Gia(USD)'] <= upper_bound)]

# Lưu tệp mới
df.to_csv('NEW_DATA.csv', index=False)
