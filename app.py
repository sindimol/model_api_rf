from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
import os

# Load model dan scaler regresi
model = joblib.load('model_regresi_random.pkl')       # Model regresi kadar air
scaler = joblib.load('scaler_regresi_rf.pkl')         # StandardScaler untuk RGB

app = Flask(__name__)

# Fungsi ekstraksi fitur RGB
def extract_rgb_features(image):
    image = image.resize((150, 150))
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        r_mean = np.mean(image_np[:, :, 0])
        g_mean = np.mean(image_np[:, :, 1])
        b_mean = np.mean(image_np[:, :, 2])
        return [r_mean, g_mean, b_mean]
    else:
        raise ValueError("Gambar harus berupa RGB (3 channel)")

@app.route('/predict_kadar_air', methods=['POST'])
def predict_kadar_air():
    try:
        image_file = request.files['image']
        image = Image.open(image_file).convert('RGB')

        # Ekstrak RGB
        rgb_features = extract_rgb_features(image)
        features = np.array(rgb_features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Prediksi kadar air
        kadar_air = model.predict(features_scaled)[0]
        kadar_air = round(float(kadar_air), 2)

        # Konversi ke status
        if kadar_air >= 70:
            status = 'Segar'
        elif kadar_air >= 50:
            status = 'Sedang'
        else:
            status = 'Kering'

        return jsonify({
            'kadar_air': kadar_air,
            'status': status
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
