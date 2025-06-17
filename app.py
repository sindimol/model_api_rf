from flask import Flask, request, jsonify
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_reg.pkl')
scaler = joblib.load('scaler_reg.pkl')

def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    levels = 256
    glcm = np.zeros((levels, levels), dtype=np.float64)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1] - 1):
            row = gray[i, j]
            col = gray[i, j + 1]
            glcm[row, col] += 1

    glcm += glcm.T
    glcm /= glcm.sum()

    contrast = np.sum([(i - j) ** 2 * glcm[i, j] for i in range(levels) for j in range(levels)])
    energy = np.sum(glcm ** 2)
    homogeneity = np.sum([glcm[i, j] / (1 + abs(i - j)) for i in range(levels) for j in range(levels)])

    return contrast, energy, homogeneity

def get_status(kadar_air):
    if kadar_air > 70:
        return "Segar"
    elif kadar_air >= 50:
        return "Sedang"
    else:
        return "Kering"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil file gambar
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Ekstrak fitur
        contrast, energy, homogeneity = extract_features(img)
        mean_R = np.mean(img[:, :, 2])
        mean_G = np.mean(img[:, :, 1])
        mean_B = np.mean(img[:, :, 0])

        fitur = np.array([[mean_R, mean_G, mean_B, contrast, energy, homogeneity]])
        fitur_scaled = scaler.transform(fitur)

        # Prediksi kadar air
        kadar_air_pred = model.predict(fitur_scaled)[0]
        status = get_status(kadar_air_pred)

        return jsonify({
            'kadar_air': round(kadar_air_pred, 2),
            'status': status
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
