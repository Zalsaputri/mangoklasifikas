import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from utils.download_dataset import download_dataset
download_dataset()
import cv2
import numpy as np
import joblib
from PIL import Image
import io
import uuid # Untuk menghasilkan nama file unik

app = Flask(__name__)

UPLOAD_FOLDER = 'uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = 100

MODEL_PATH = 'models/svm_model_mangga.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
# SCALER_PATH tidak ada di versi ini

# Muat model dan encoder saat aplikasi dimulai
model = None
le = None
# scaler = None # Scaler tidak dimuat di versi ini
try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    print("Model dan LabelEncoder berhasil dimuat.")
except FileNotFoundError:
    print(f"Error: Pastikan file '{MODEL_PATH}' dan '{LABEL_ENCODER_PATH}' ada di folder 'models'.")
    print("Silakan jalankan 'traintest.py' terlebih dahulu untuk melatih dan menyimpan model.")
except Exception as e:
    print(f"Error tidak terduga saat memuat model/encoder: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploaded_images/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Tidak ada file yang diunggah.'), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error='Tidak ada file yang dipilih.'), 400
        
        # Di versi ini, kita hanya butuh model dan le, tidak perlu scaler
        if file and model and le: 
            try:
                filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_url = f'/uploaded_images/{filename}'

                img_pil = Image.open(filepath).convert("L") 
                img_cv2_grayscale = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE)))

                # >>>>>> Tidak ada Ekstraksi Fitur LBP atau Penskalaan di sini <<<<<<
                # Fitur adalah pixel values mentah yang diratakan
                features = img_cv2_grayscale.flatten()
                features = features.reshape(1, -1) # Reshape untuk input model

                # DEBUGGING: Print jumlah fitur yang dihasilkan di app.py
                print(f"DEBUG APP.PY: Fitur dihasilkan (Pixel values): {features.shape[1]} fitur.")
                
                # Melakukan prediksi
                pred_label_idx = model.predict(features)[0] 
                pred_proba_scores = model.decision_function(features)
                
                exp_scores = np.exp(pred_proba_scores - np.max(pred_proba_scores))
                probabilities = exp_scores / np.sum(exp_scores)
                confidence_percentage = np.max(probabilities) * 100 

                pred_label = le.inverse_transform([pred_label_idx])[0]

                return render_template('results.html', 
                                       image_url=image_url, 
                                       prediction=pred_label, 
                                       confidence=round(confidence_percentage, 2))

            except Exception as e:
                print(f"Error saat memproses gambar atau prediksi: {e}")
                return render_template('index.html', error=f'Gagal memproses gambar: {e}'), 500
        else:
            return render_template('index.html', error='Model atau LabelEncoder belum dimuat dengan benar. Periksa log server aplikasi.'), 500
    
    return render_template('index.html', error='Metode tidak diizinkan. Mohon gunakan POST untuk upload.'), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
