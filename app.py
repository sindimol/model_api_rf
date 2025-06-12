from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
import os

# Load model dan scaler
model = joblib.load('model_rf.pkl')         # Model klasifikasi
scaler = joblib.load('scaler.pkl')          # StandardScaler saat training
label_encoder = joblib.load('label_encoder.pkl')  # Encoder label, jika ada

app = Flask(__name__)

# Mapping status ke estimasi kadar air (rata-rata)
kadar_air_mapping = {
    'Segar': 80.0,
    'Sedang': 65.0,
    'Kering': 45.0
}

# Fungsi ekstraksi RGB mean
def extract_rgb_features(image):
    image = image.resize((150, 150))  # Ukuran lebih besar agar lebih variatif
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

        # Ekstrak fitur RGB
        rgb_features = extract_rgb_features(image)
        features = np.array(rgb_features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Prediksi label
        prediction = model.predict(features_scaled)[0]
        if hasattr(label_encoder, "inverse_transform"):
            original_label = label_encoder.inverse_transform([prediction])[0]
        else:
            original_label = str(prediction)

        # Ambil status akhir (Segar/Sedang/Kering saja)
        if 'segar' in original_label:
            status = 'Segar'
        elif 'sedang' in original_label:
            status = 'Sedang'
        elif 'kering' in original_label:
            status = 'Kering'
        else:
            status = 'Tidak Diketahui'

        kadar_air = kadar_air_mapping.get(status, 0.0)

        return jsonify({
            'klasifikasi': status,
            'kadar_air': kadar_air
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
