import cv2

# Memuat model Haar Cascade yang disediakan oleh OpenCV.
lokasi_model = "haarcascade_frontalface_default.xml"
detektor = cv2.CascadeClassifier(lokasi_model)

kamera = cv2.VideoCapture(0)
if not kamera.isOpened():
    raise SystemExit("Kamera tidak dapat dibuka. Periksa izin kamera.")

print("Tekan Q pada jendela video untuk keluar.")

while True:
    berhasil, gambar = kamera.read()
    if not berhasil:
        print("Gambar kamera gagal dibaca.")
        break

    # Haar Cascade bekerja pada citra grayscale.
    abu = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)

    # Cari area yang kemungkinan merupakan wajah.
    wajah = detektor.detectMultiScale(
        abu,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(70, 70)
    )

    # Gambar kotak di setiap wajah yang terdeteksi.
    for (x, y, lebar, tinggi) in wajah:
        cv2.rectangle(
            gambar,
            (x, y),
            (x + lebar, y + tinggi),
            (0, 165, 255),
            2
        )

    # Tampilkan jumlah wajah.
    cv2.putText(
        gambar,
        f"Jumlah wajah: {len(wajah)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 165, 255),
        2
    )

    cv2.imshow("Deteksi Wajah", gambar)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()