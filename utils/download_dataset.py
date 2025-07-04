import os
import zipfile
import requests

def download_dataset():
    if os.path.exists("mangga_terbaru"):
        print("✅ Dataset sudah tersedia.")
        return

    print("📥 Mendownload dataset...")
    file_id = "176IXHZYsONcoBPemWQ0zkom1nAhRdAzl"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)

    zip_path = "mangga_terbaru.zip"
    with open(zip_path, "wb") as f:
        f.write(r.content)

    print("📂 Mengekstrak dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    print("✅ Dataset siap digunakan.")
