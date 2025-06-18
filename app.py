import joblib
import pandas as pd
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Muat model dan scaler
try:
    loaded_model = joblib.load("model_kuat_tekan_28_hari.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    print("Error: Pastikan 'model_kuat_tekan_28_hari.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    exit() # Keluar jika file tidak ditemukan

# Definisikan fitur-fitur yang digunakan oleh model (sesuai urutan training)
FEATURES = [
    'Water', 'Cement Type I', 'HYDRAULIC CEMENT', 'FLY ASH (F)',
    'SPLIT 10 - 20', 'M-SAND', 'DARATARD 36 MR', 'DARACEM 133 ADV',
    'Plastiment 923', 'Viscocrete 8050'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data_input = {}
        for feature in FEATURES:
            try:
                data_input[feature] = float(request.form[feature])
            except ValueError:
                return render_template('result.html', prediction="Input tidak valid. Pastikan semua nilai adalah angka.", is_error=True)

        # Konversi ke DataFrame
        data_baru_df = pd.DataFrame([data_input])

        # Preprocessing: Standarisasi fitur
        data_baru_df_scaled = loaded_scaler.transform(data_baru_df)

        # Prediksi
        prediksi_kuat_tekan = loaded_model.predict(data_baru_df_scaled)[0]

        # Siapkan data untuk visualisasi Plotly
        # Karena ini hanya 1 prediksi, kita bisa membuat bar chart atau gauge.
        # Contoh sederhana: bar chart dengan 1 bar
        chart_data = [
            {
                'x': ['Kuat Tekan Beton'],
                'y': [prediksi_kuat_tekan],
                'type': 'bar',
                'name': 'Kuat Tekan (MPa)',
                'marker': {
                    'color': '#c3d500',
                    'line': {'width': 2, 'color': '#5a5a5a'},
                    'opacity': 0.95
                },
                'hoverinfo': 'y',
                'text': [f"{prediksi_kuat_tekan:.2f} MPa"],
                'textposition': 'auto',
                'width': 0.5
            }
        ]
        chart_layout = {
            'title': 'Prediksi Kuat Tekan Beton 28 Hari',
            'yaxis': {'title': 'Kuat Tekan (MPa)', 'range': [0, 50], 'gridcolor': '#e0e0e0'},
            'xaxis': {'showticklabels': False},
            'plot_bgcolor': '#f8f9fa',
            'paper_bgcolor': '#f8f9fa',
            'font': {'color': '#5a5a5a', 'family': 'Segoe UI, Arial, sans-serif'},
            'bargap': 0.4,
        }


        return render_template('result.html',
                               prediction=f"{prediksi_kuat_tekan:.2f}",
                               chart_data=chart_data,
                               chart_layout=chart_layout)

if __name__ == '__main__':
    app.run(debug=True)