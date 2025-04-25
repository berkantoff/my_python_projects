import cv2

# Kamerayı başlat (0: varsayılan kamera)
kamera = cv2.VideoCapture(0)

# Kamera açıldı mı kontrol et
if not kamera.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, kare = kamera.read()
    if not ret:
        print("Kare alınamadı.")
        break

    cv2.imshow('Kamera', kare)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
