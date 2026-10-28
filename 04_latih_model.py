import cv2 
import numpy as np 
from pathlib import Path 
  
# LBPH tersedia pada paket opencv-contrib-python. 
if not hasattr(cv2, "face"): 
    raise SystemExit( 
        "Modul cv2.face tidak tersedia. Aktifkan virtual environment " 
        "dan instal opencv-contrib-python." 
    ) 
  
folder = Path("data_wajah") / "pengguna_01" 
if not folder.exists(): 
    raise SystemExit( 
        "Folder data tidak ditemukan. Jalankan 03_registrasi_wajah.py dahulu." 
    ) 
  
gambar_latih = [] 
label_latih = [] 
  
for file_gambar in folder.iterdir(): 
    if file_gambar.suffix.lower() not in (".jpg", ".jpeg", ".png"): 
        continue 
  
    gambar = cv2.imread(str(file_gambar), cv2.IMREAD_GRAYSCALE) 
    if gambar is None: 
        continue 
  
    gambar = cv2.resize(gambar, (200, 200)) 
    gambar_latih.append(gambar) 
    label_latih.append(1)  # Label 1 berarti kelas pengguna_01. 
  
if len(gambar_latih) < 2: 
    raise SystemExit("Sampel kurang dari dua. Jalankan ulang registrasi.") 
  
# Buat model LBPH, latih menggunakan gambar dan label. 
model = cv2.face.LBPHFaceRecognizer_create() 
model.train(gambar_latih, np.array(label_latih, dtype=np.int32)) 
  
# Simpan model agar dapat digunakan program verifikasi. 
model.write("model_wajah.yml") 
print("Pelatihan selesai.") 
print("Jumlah gambar latihan:", len(gambar_latih)) 
print("Model disimpan sebagai model_wajah.yml") 