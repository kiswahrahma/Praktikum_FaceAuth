import cv2 
from pathlib import Path 
  
# Gunakan ID samaran, bukan NIK, NIM, atau nama lengkap. 
ID_PENGGUNA = "pengguna_01" 
TARGET_SAMPEL = 30 
  
folder = Path("data_wajah") / ID_PENGGUNA 
folder.mkdir(parents=True, exist_ok=True) 
  
detektor = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
kamera = cv2.VideoCapture(0) 
  
if not kamera.isOpened(): 
    raise SystemExit("Kamera tidak dapat dibuka.") 
  
jumlah = 0 
print("Posisikan wajah di depan kamera.") 
print("Gerakkan kepala sedikit secara alami agar sampel bervariasi.") 
print("Tekan Q untuk membatalkan.") 
  
while jumlah < TARGET_SAMPEL: 
    berhasil, gambar = kamera.read() 
    if not berhasil: 
        print("Gagal membaca gambar kamera.") 
        break 
  
    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY) 
    daftar_wajah = detektor.detectMultiScale( 
        abu, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80) 
    ) 
  
    if len(daftar_wajah) > 0: 
        # Jika ada lebih dari satu wajah, pilih yang kotaknya terbesar. 
        x, y, w, h = max( 
            daftar_wajah, key=lambda kotak: kotak[2] * kotak[3] 
        ) 
        potongan = abu[y:y+h, x:x+w] 
        potongan = cv2.resize(potongan, (200, 200)) 
  
        # Simpan sampel dengan nama berurutan. 
        jumlah += 1 
        nama_file = folder / f"sampel_{jumlah:03d}.jpg" 
        cv2.imwrite(str(nama_file), potongan) 
  
        cv2.rectangle(gambar, (x, y), (x+w, y+h), (0, 165, 255), 2) 
  
    cv2.putText( 
        gambar, f"Sampel: {jumlah}/{TARGET_SAMPEL}", (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2 
    ) 
    cv2.imshow("Registrasi Wajah", gambar) 
  
    if cv2.waitKey(150) & 0xFF == ord("q"): 
        break 
  
kamera.release() 
cv2.destroyAllWindows() 
print(f"Jumlah sampel tersimpan: {jumlah}") 
print(f"Lokasi penyimpanan: {folder}") 