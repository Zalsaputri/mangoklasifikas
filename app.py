from flask import Flask, render_template, request, jsonify
import os
from utils.download_dataset import download_dataset
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image
import joblib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Buat folder upload jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model dan label encoder
try:
    model = joblib.load('models/mango_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    print("Model dan LabelEncoder berhasil dimuat.")
except Exception as e:
    print("Error tidak terduga saat memuat model/encoder:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'Tidak ada file yang diunggah', 400

    file = request.files['file']
    if file.filename == '':
        return 'Nama file kosong', 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # Baca gambar dan ubah ukuran
        image = Image.open(filepath).convert('RGB')
        image = image.resize((100, 100))  # Sesuaikan dengan input model
        image_array = np.array(image).flatten().reshape(1, -1)

        prediction = model.predict(image_array)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template('index.html', prediction=predicted_label, image_path=filepath)
    except Exception as e:
        return f'Error saat memproses gambar: {e}', 500

@app.route('/download_dataset')
def download_dataset_route():
    try:
        download_dataset()
        return jsonify({'message': 'Dataset berhasil didownload dan diekstrak.'})
    except Exception as e:
        return jsonify({'error': f'Gagal download dataset: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
