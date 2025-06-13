from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# Inisialisasi Flask app
app = Flask(__name__)

# Load model, scaler, dan label encoder
model = load_model('model_ann.h5')
scaler = joblib.load('scaler_ann.pkl')
label_encoder = joblib.load('label_encoder_ann.pkl')

# Fungsi ekstraksi fitur RGB
def extract_features(image):
    image = image.resize((150, 150))
    image_np = np.array(image)

    if image_np.ndim == 3 and image_np.shape[2] == 3:
        r_mean = np.mean(image_np[:, :, 0])
        g_mean = np.mean(image_np[:, :, 1])
        b_mean = np.mean(image_np[:, :, 2])
        return [r_mean, g_mean, b_mean]
    else:
        raise ValueError("Gambar tidak valid atau tidak RGB.")

@app.route('/predict_kualitas_cabai', methods=['POST'])
def predict_kualitas():
    try:
        image_file = request.files['image']
        kadar_air = float(request.form['kadar_air'])  # Dikirim dari Flutter juga

        image = Image.open(image_file).convert('RGB')
        rgb_features = extract_features(image)

        fitur_input = np.array(rgb_features + [kadar_air]).reshape(1, -1)
        fitur_scaled = scaler.transform(fitur_input)

        pred_prob = model.predict(fitur_scaled)[0]
        pred_class_index = np.argmax(pred_prob)
        pred_label = label_encoder.inverse_transform([pred_class_index])[0]

        return jsonify({
            'prediksi': pred_label,
            'confidence': round(float(np.max(pred_prob)) * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
