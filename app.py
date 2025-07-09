from flask import Flask, request, render_template, send_from_directory
import os
import uuid
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Maksimal 5 MB

# Pastikan folder upload tersedia
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model & encoder
model = joblib.load('models/svm_model_mangga.pkl') 
label_encoder = joblib.load('models/label_encoder.pkl')    
IMG_SIZE = 100

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file di request.')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='Tidak ada file dipilih.')

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

        # Kembalikan ke index.html dengan hasil prediksi
        return render_template('index.html', prediction=label, confidence=round(confidence, 2))

    except Exception as e:
        return render_template('index.html', error=f"Gagal memproses gambar: {e}")

@app.route('/uploaded_images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
