import cv2 
  
if not hasattr(cv2, "face"): 
    raise SystemExit("Modul cv2.face tidak tersedia. Instal opencv-contrib-python.") 
  
model = cv2.face.LBPHFaceRecognizer_create() 
try: 
    model.read("model_wajah.yml") 
except cv2.error: 
    raise SystemExit("Model tidak ditemukan. Jalankan 04_latih_model.py dahulu.") 
  
detektor = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
kamera = cv2.VideoCapture(0) 
if not kamera.isOpened(): 
    raise SystemExit("Kamera tidak dapat dibuka.") 
  
# Threshold ini hanya nilai awal untuk eksperimen kelas. 
# Skor LBPH adalah jarak, bukan probabilitas. 
THRESHOLD = 60 
  
print("Tekan Q pada jendela video untuk keluar.") 
print("Uji dengan pengguna terdaftar dan relawan yang menyetujui.") 
  
while True: 
    berhasil, gambar = kamera.read() 
    if not berhasil: 
        break 
  
    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY) 
    daftar_wajah = detektor.detectMultiScale( 
        abu, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80) 
    ) 
  
    pesan = "Wajah tidak terdeteksi" 
    warna = (0, 165, 255) 
  
    if len(daftar_wajah) > 0: 
        x, y, w, h = max( 
            daftar_wajah, key=lambda kotak: kotak[2] * kotak[3] 
        ) 
        potongan = cv2.resize(abu[y:y+h, x:x+w], (200, 200)) 
  
        label, skor_jarak = model.predict(potongan) 
  
        if label == 1 and skor_jarak < THRESHOLD: 
            pesan = "COCOK (indikasi prototipe)" 
            warna = (0, 150, 0) 
        else: 
            pesan = "TIDAK COCOK" 
            warna = (0, 0, 255) 
  
        cv2.rectangle(gambar, (x, y), (x+w, y+h), warna, 2) 
        cv2.putText( 
            gambar, f"Skor jarak: {skor_jarak:.1f}", (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna, 2 
        )
        cv2.putText(
            gambar, pesan, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2
        )
        cv2.imshow("Verifikasi Wajah", gambar) 
        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break

kamera.release() 
cv2.destroyAllWindows() 