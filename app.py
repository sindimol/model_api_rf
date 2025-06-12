from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
import os

# Load model regresi dan scaler
model = joblib.load('model_regresi_rgb.pkl')
scaler = joblib.load('scaler_regresi_rgb.pkl')

app = Flask(__name__)

# Ekstraksi fitur RGB
def extract_rgb_features(image):
    image = image.resize((100, 100))
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
        image = Image.open(image_file).convert('RGB')
        rgb_features = extract_rgb_features(image)
        features_scaled = scaler.transform([rgb_features])
        predicted_kadar_air = model.predict(features_scaled)[0]

        # Klasifikasi kondisi
        if predicted_kadar_air >= 80:
            status = "Segar"
        elif predicted_kadar_air >= 60:
            status = "Sedang"
        else:
            status = "Kering"

        return jsonify({
            'kadar_air': round(float(predicted_kadar_air), 2),
            'status': status
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
