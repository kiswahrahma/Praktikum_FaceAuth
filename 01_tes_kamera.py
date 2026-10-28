import cv2 
  
# 1. Buka kamera default (biasanya webcam laptop). 
kamera = cv2.VideoCapture(0) 
  
# 2. Periksa apakah kamera berhasil dibuka. 
if not kamera.isOpened(): 
    print("Kamera gagal dibuka.") 
    print("Periksa izin kamera Windows atau tutup aplikasi lain yang memakai kamera.") 
    raise SystemExit 
  
print("Kamera aktif. Tekan Q pada jendela video untuk keluar.") 
  
# 3. Baca dan tampilkan video terus-menerus. 
while True: 
    berhasil, gambar = kamera.read() 
  
    # Jika gambar gagal dibaca, hentikan loop. 
    if not berhasil: 
        print("Gagal membaca gambar dari kamera.") 
        break 
  
    # Tampilkan gambar kamera dalam jendela. 
    cv2.imshow("Tes Kamera", gambar) 
  
    # Tekan Q untuk keluar. 
    if cv2.waitKey(1) & 0xFF == ord("q"): 
        break 
  
# 4. Lepaskan kamera dan tutup jendela. 
kamera.release() 
cv2.destroyAllWindows() 