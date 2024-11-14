from flask import Flask, render_template, request
import joblib  # Import thư viện joblib để load mô hình
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import pandas as pd

app = Flask(__name__)

# Load mô hình đã huấn luyện
trained_model = joblib.load('trained_model.joblib')

# Load dữ liệu
data = pd.read_csv('NEW_DATA_UPDATE.csv')

# Phân loại cột
numeric_columns = ['So phong ngu', 'So phong tam', 'Dien tich(feet)']
categorical_columns = ['Loai', 'Khu vuc', 'Duong']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận giá trị từ form
    khu_vuc = request.form['khu_vuc']
    duong = request.form['duong']
    so_phong_ngu = int(request.form['so_phong_ngu'])
    so_phong_tam = int(request.form['so_phong_tam'])
    dien_tich = float(request.form['dien_tich'])
    loai = request.form['loai']
    price_range = request.form['price_range']  # Nhận phạm vi giá người dùng chọn

    # Tạo dữ liệu quan sát mới
    new_observation = pd.DataFrame({
        'Khu vuc': [khu_vuc],
        'Duong': [duong],
        'So phong ngu': [so_phong_ngu],
        'So phong tam': [so_phong_tam],
        'Dien tich(feet)': [dien_tich],
        'Loai': [loai]
    })

    # Dự đoán giá trị dựa trên mô hình
    predicted_price = trained_model.predict(new_observation)

    # Quyết định dựa trên giá trị dự đoán và phạm vi giá
    decision = "Nên mua" if predicted_price <= 2000 else "Không mua"

    # Trả về kết quả dự đoán
    return render_template('result.html', predicted_price=predicted_price[0], decision=decision, price_range=price_range)

if __name__ == '__main__':
    app.run(debug=True)
