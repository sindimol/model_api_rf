from flask import Flask, request, jsonify
import os
import numpy as np
import joblib
from PIL import Image

# Load model regresi dan scaler
model = joblib.load('model_regresi_rgb.pkl')
scaler = joblib.load('scaler_regresi_rgb.pkl')

app = Flask(__name__)

# Fungsi ekstraksi fitur mean R, G, B dari gambar
def extract_rgb_features(image):
    image = image.resize((100, 100))  # Ukuran seragam
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        r_mean = np.mean(image_np[:, :, 0])
        g_mean = np.mean(image_np[:, :, 1])
        b_mean = np.mean(image_np[:, :, 2])
        return [r_mean, g_mean, b_mean]
    else:
        raise ValueError("Gambar harus RGB (3 channel)")

@app.route('/predict_kadar_air', methods=['POST'])
def predict_kadar_air():
    try:
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')  # Konversi ke RGB

        # Simpan gambar untuk debugging
        image.save("last_uploaded.jpg")
        print("Gambar disimpan sebagai 'last_uploaded.jpg'")

        rgb_features = extract_rgb_features(image)
        print("RGB Features (mean):", rgb_features)

        # Validasi: deteksi jika gambar terlalu terang atau gelap
        if any([x < 5 or x > 250 for x in rgb_features]):
            print("Gambar tidak valid: terlalu terang atau gelap")
            return jsonify({
                'kadar_air': None,
                'klasifikasi': 'Gambar tidak valid atau bukan cabai'
            })

        # Prediksi
        features_scaled = scaler.transform([rgb_features])
        print("Scaled Features:", features_scaled)

        predicted_kadar_air = model.predict(features_scaled)[0]
        print("Predicted Kadar Air:", predicted_kadar_air)

        # Klasifikasi berdasarkan kadar air
        if predicted_kadar_air >= 80:
            kondisi = "Segar"
        elif predicted_kadar_air >= 60:
            kondisi = "Sedang"
        else:
            kondisi = "Kering"

        return jsonify({
            'kadar_air': round(float(predicted_kadar_air), 2),
            'klasifikasi': kondisi
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
