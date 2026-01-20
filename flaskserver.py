from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

print("Sistem başlatılıyor...")

# Modeli yüklemeye çalışıyoruz
model = None
scaler = None

# Dosyalar yerinde mi kontrol edelim
if os.path.exists('cnn_model.h5'):
    model = load_model('cnn_model.h5')
    print("BASARILI: CNN Modeli yüklendi.")
else:
    print("HATA: 'cnn_model.h5' dosyası bulunamadı!")

if os.path.exists('scaler.pkl'):
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    print("BASARILI: Scaler yüklendi.")
else:
    print("HATA: 'scaler.pkl' dosyası bulunamadı!")


# İŞTE EKSİK OLAN KISIM BURASIYDI:
@app.route('/', methods=['GET'])
def home():
    if model and scaler:
        return "<h1>TEBRİKLER! CNN Modeli ve API Sorunsuz Çalışıyor! 🚀</h1>"
    else:
        return "<h1>Model dosyaları yüklenemedi. Lütfen model_egit.py dosyasını çalıştırdığından emin ol.</h1>"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Basit hata kontrolleri
        if not model or not scaler:
            return jsonify({'error': 'Model sunucuda yüklü değil.'}), 500

        if 'features' not in data or not isinstance(data['features'], list):
            return jsonify({'error': 'Hatalı format. "features" listesi gönderilmeli.'}), 400

        # Veriyi hazırla
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_cnn = features_scaled.reshape(1, 30, 1)

        # Tahmin yap
        tahmin_olasiligi = model.predict(features_cnn)[0][0]

        # Sonuç
        sonuc = "Kötü Huylu (Kanser Riski)" if tahmin_olasiligi < 0.5 else "İyi Huylu (Temiz)"

        return jsonify({
            'prediction': sonuc,
            'probability': float(tahmin_olasiligi)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)