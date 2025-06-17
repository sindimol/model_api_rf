from flask import Flask, request, jsonify
import joblib

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Load model yang sudah disimpan
model = joblib.load('model_rf.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mendapatkan input dari form yang dikirimkan
        kadar_air = float(request.form['kadar_air'])  # Dikirim dari Flutter
        fitur = [kadar_air]  # Kamu bisa menambah fitur lain sesuai kebutuhan

        # Melakukan prediksi
        prediksi = model.predict([fitur])[0]

        return jsonify({
            'prediksi': prediksi
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
