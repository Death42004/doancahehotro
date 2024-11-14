import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('NEW_DATA.csv')

# Xác định các cột số và cột phân loại
numeric_columns = ['Gia(USD)', 'So phong ngu', 'So phong tam', 'Dien tich(feet)', 'Kinh do', 'Vi do']
categorical_columns = ['Loai', 'Khu vuc', 'Duong']

# Hàm vẽ biểu đồ phân phối và biểu đồ phân tán với đường hồi quy
def num_combined_plot(data, x, y):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Vẽ biểu đồ phân phối (histogram + KDE)
    sns.histplot(data=data, x=x, kde=True, ax=axes[0], color='coral')
    
    # Vẽ biểu đồ phân tán với đường hồi quy
    sns.regplot(data=data, x=x, y=y, ax=axes[1], color='teal',
                scatter_kws={'edgecolor': 'white'}, line_kws={"color": "coral"})
    
    # Tính hệ số tương quan
    corr_coeff = data[[x, y]].corr().iloc[0, 1]
    
    # Chú thích hệ số tương quan trên biểu đồ phân tán
    axes[1].annotate(f'Correlation : {corr_coeff:.2f}', xy=(0.65, 0.9), xycoords='axes fraction',
                     fontsize=14, color='coral')

    # Tinh chỉnh giao diện biểu đồ
    sns.despine(bottom=True, left=True)
    axes[0].set(xlabel=f'{x}', ylabel='Frequency', title=f'{x} Distribution')
    axes[1].set(xlabel=f'{x}', ylabel=f'{y}', title=f'{x} vs {y}')
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()

    plt.tight_layout()  # Điều chỉnh để subplots không bị chồng chéo
    plt.show()

# Hàm tạo lưới các đồ thị (countplot, boxplot, scatterplot)
def create_subplot_grid(data, x, y):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Vẽ biểu đồ countplot theo tỷ lệ phần trăm
    sns.countplot(data=data, x=x, hue=x, ax=axes[0], palette='Set2')
    axes[0].set(title=f'{x} Frequency')
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].set_ylabel('Count (%)')
    
    # Tính toán và chú thích tỷ lệ phần trăm
    total = len(data)
    for p in axes[0].patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        x_ = p.get_x() + p.get_width() / 2
        y_ = p.get_height()
        axes[0].annotate(percentage, (x_, y_), ha='center', va='bottom')
    
    # Vẽ biểu đồ boxplot
    sns.boxplot(data=data, x=x, y=y, ax=axes[1], hue=x, palette='Set2')
    axes[1].set(title=f'Price vs. {x}')
    axes[1].tick_params(axis='x', rotation=90)
    
    # Vẽ biểu đồ phân tán
    sns.scatterplot(data=data, x=x, y=y, ax=axes[2], hue=x, palette='Set2')
    axes[2].set(title=f'{y} vs. {x}')
    axes[2].tick_params(axis='x', rotation=90)
    axes[2].yaxis.set_label_position("right")
    
    # Thêm đường hồi quy vào biểu đồ phân tán
    sns.regplot(data=data, x=x, y=y, ax=axes[2], color='coral', scatter=False)
    axes[2].get_legend().remove()

    plt.tight_layout()  # Điều chỉnh để không bị chồng chéo
    plt.show()

# Vẽ pairplot cho các biến số
sns.pairplot(data[numeric_columns])
plt.suptitle('Pairplot for num_features', y=1.02)
plt.show()

# Tạo biểu đồ heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data[numeric_columns].corr(), annot=True, cmap='Spectral', linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap', fontsize=15)
plt.show()

# Trực quan dữ liệu diện tích vs giá
num_combined_plot(data, 'Dien tich(feet)', 'Gia(USD)')

# Trực quan dữ liệu số phòng ngủ vs giá
create_subplot_grid(data, 'So phong ngu', 'Gia(USD)')

# Trực quan dữ liệu phòng tắm với giá
create_subplot_grid(data, 'So phong tam', 'Gia(USD)')

# Trực quan vị trí (Kinh độ vs Vĩ độ)
fig, ax = plt.subplots()
scatter = ax.scatter(data['Kinh do'], data['Vi do'], c=data['Gia(USD)'], cmap='viridis')
ax.set_xlabel('Kinh do')
ax.set_ylabel('Vi do')
fig.colorbar(scatter)
plt.show()
