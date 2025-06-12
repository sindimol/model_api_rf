from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image

# Load model dan scaler
model = joblib.load('model_rf.pkl')  # model klasifikasi
scaler = joblib.load('scaler.pkl')   # StandardScaler yang dipakai saat training

app = Flask(__name__)

# Mapping label hasil model ke status umum
label_mapping = {
    'hijau_segar': 'Segar',
    'merah_segar': 'Segar',
    'hijau_sedang': 'Sedang',
    'merah_sedang': 'Sedang',
    'hijau_kering': 'Kering',
    'merah_kering': 'Kering'
}

# Fungsi ekstraksi fitur dari gambar
def extract_rgb_features(image):
    image = image.resize((100, 100))
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        r_mean = np.mean(image_np[:, :, 0])
        g_mean = np.mean(image_np[:, :, 1])
        b_mean = np.mean(image_np[:, :, 2])
        return [r_mean, g_mean, b_mean]
    else:
        raise ValueError("Gambar harus memiliki 3 channel RGB")

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        image_file = request.files['image']
        kadar_air_str = request.form.get('kadar_air', None)

        if kadar_air_str is None:
            return jsonify({'error': 'Parameter kadar_air tidak ditemukan'}), 400

        kadar_air = float(kadar_air_str) / 100.0  # ubah ke skala 0–1
        image = Image.open(image_file).convert('RGB')
        rgb_features = extract_rgb_features(image)

        # Gabungkan RGB dan kadar air
        features = np.array(rgb_features + [kadar_air]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Prediksi kelas
        prediction = model.predict(features_scaled)
        original_label = str(prediction[0])
        mapped_label = label_mapping.get(original_label, "Tidak Diketahui")

        return jsonify({
            'klasifikasi': mapped_label,
            'label_model': original_label
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
