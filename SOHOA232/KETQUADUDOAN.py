import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib

# Định nghĩa các cột số và cột phân loại
numeric_columns = ['So phong ngu', 'So phong tam', 'Dien tich(feet)']
categorical_columns = ['Loai', 'Khu vuc', 'Duong']

# Định nghĩa bộ tiền xử lý dữ liệu
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_columns),  # Chuẩn hóa cột số
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # Mã hóa cột phân loại
    ])

# Tạo pipeline với bộ tiền xử lý và mô hình hồi quy
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Giả định dữ liệu nằm trong 'NEW_DATA.csv'
data = pd.read_csv('NEW_DATA_UPDATE.csv')

# Chia dữ liệu thành biến đầu vào (features) và biến mục tiêu (target)
X = data[numeric_columns + categorical_columns]
y = data['Gia(USD)']

# Huấn luyện mô hình
model.fit(X, y)

# Lưu mô hình đã huấn luyện
joblib.dump(model, 'trained_model.joblib')
