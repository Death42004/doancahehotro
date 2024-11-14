from flask import Flask, render_template, request
import joblib  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

app = Flask(__name__)
trained_model = joblib.load('trained_model.joblib')

data = pd.read_csv('NEW_DATA_UPDATE.csv')
numeric_columns = ['So phong ngu', 'So phong tam', 'Dien tich(feet)']
categorical_columns = ['Loai', 'Khu vuc', 'Duong']

unique_khu_vuc = data['Khu vuc'].unique().tolist()
unique_duong = data['Duong'].unique().tolist()
unique_so_phong_ngu = sorted(data['So phong ngu'].unique().tolist())
unique_so_phong_tam = sorted(data['So phong tam'].unique().tolist())
unique_dien_tich = sorted(data['Dien tich(feet)'].unique().tolist())
unique_loai = data['Loai'].unique().tolist()

@app.route('/')
def home():
    return render_template('index.html', unique_khu_vuc=unique_khu_vuc, unique_duong=unique_duong,
                           unique_so_phong_ngu=unique_so_phong_ngu, unique_so_phong_tam=unique_so_phong_tam,
                           unique_dien_tich=unique_dien_tich, unique_loai=unique_loai)

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận thông tin từ form
    khu_vuc = request.form['khu_vuc']
    duong = request.form['duong']
    so_phong_ngu = int(request.form['so_phong_ngu'])
    so_phong_tam = int(request.form['so_phong_tam'])
    dien_tich = float(request.form['dien_tich'])
    loai = request.form['loai']
    price_range = request.form['price_range']

    # Dữ liệu đầu vào mới
    new_observation = pd.DataFrame({
        'Khu vuc': [khu_vuc],
        'Duong': [duong],
        'So phong ngu': [so_phong_ngu],
        'So phong tam': [so_phong_tam],
        'Dien tich(feet)': [dien_tich],
        'Loai': [loai]
    })

    # Dự đoán giá trị dựa trên mô hình
    predicted_price = trained_model.predict(new_observation)[0]

    # Chuyển đổi price_range thành giá trị số để so sánh
    if price_range == "0-20000":
        min_price, max_price = 0, 2000
    elif price_range == "20000-50000":
        min_price, max_price = 2000, 10000
    elif price_range == "50000-100000":
        min_price, max_price = 10000, 50000
    elif price_range == "100000-300000":
        min_price, max_price = 50000, 100000
    elif price_range == "500000+":
        min_price, max_price = 100000, float('inf')
    else:
        min_price, max_price = 0, float('inf')

    # Đưa ra quyết định dựa trên khoảng giá
    if min_price <= predicted_price <= max_price:
        decision = "Khuyến nghị mua - Giá nằm trong khoảng yêu cầu"
    else:
        decision = "Không mua - Giá ngoài khoảng yêu cầu"

    # Trả về kết quả dự đoán và quyết định
    return render_template('result.html', predicted_price=predicted_price, decision=decision)

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
