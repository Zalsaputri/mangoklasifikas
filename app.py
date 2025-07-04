import os
import uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
from PIL import Image
import numpy as np
import joblib
import cv2
from utils.download_dataset import download_dataset

app = Flask(__name__)

# ====== Konfigurasi ======
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = 'models/svm_model_mangga.pkl'
LABEL_PATH = 'models/label_encoder.pkl'
IMG_SIZE = 100

# ====== Load Model ======
model = None
le = None

try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_PATH)
    print("✅ Model dan LabelEncoder berhasil dimuat.")
except Exception as e:
    print(f"❌ Gagal memuat model: {e}")

# ====== Halaman Utama ======
@app.route('/')
def index():
    return render_template('index.html')

# ====== Endpoint Prediksi ======
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or file := request.files['file']:
        if file.filename == '':
            return render_template('index.html', error='Tidak ada file dipilih.'), 400
        
        try:
            # Simpan file
            filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = f'/uploaded_images/{filename}'

            # Proses gambar
            img = Image.open(filepath).convert("L").resize((IMG_SIZE, IMG_SIZE))
            features = np.array(img).flatten().reshape(1, -1)

            # Prediksi
            pred_idx = model.predict(features)[0]
            scores = model.decision_function(features)
            probs = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
            confidence = np.max(probs) * 100
            label = le.inverse_transform([pred_idx])[0]

            return render_template('index.html',
                                   image_url=image_url,
                                   prediction=label,
                                   confidence=round(confidence, 2))
        except Exception as e:
            return render_template('index.html', error=f"Gagal memproses gambar: {e}")
    return render_template('index.html', error='Tidak ada file yang valid.'), 400

# ====== Tampilkan Gambar Upload ======
@app.route('/uploaded_images/<filename>')
def uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ====== Endpoint Manual Download Dataset ======
@app.route('/download_dataset')
def manual_download_dataset():
    try:
        download_dataset()
        return jsonify({'message': 'Dataset berhasil diunduh dan diekstrak.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== Jalankan ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
