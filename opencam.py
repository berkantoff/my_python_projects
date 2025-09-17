import cv2
kamera = cv2.VideoCapture(0)
if not kamera.isOpened():
    print("Kamera açılamadı.")
    exit()
while True:
    ret, kare = kamera.read()
    if not ret:
        print("Kare alınamadı.")
        break
    cv2.imshow('Kamera', kare)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()
