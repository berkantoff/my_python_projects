import cv2
import numpy as np

def detect_target_realtime():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam başlatıldı. Çıkmak için 'q' tuşuna basın.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamıyor!")
            break
        
        # Görüntüyü işle
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Kırmızı renk aralığı (HSV'de kırmızı 2 aralıkta)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Beyaz renk aralığı
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        
        # Renk maskeleri
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Gürültü temizleme
        kernel = np.ones((5,5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        
        # Konturları bul
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # En büyük konturları bul
        if len(red_contours) > 0 and len(white_contours) > 0:
            red_contour = max(red_contours, key=cv2.contourArea)
            white_contour = max(white_contours, key=cv2.contourArea)
            
            red_area = cv2.contourArea(red_contour)
            white_area = cv2.contourArea(white_contour)
            
            # Minimum alan kontrolü
            if red_area > 1000 and white_area > 500:
                # Daireleri tespit et
                red_circle = cv2.minEnclosingCircle(red_contour)
                white_circle = cv2.minEnclosingCircle(white_contour)
                
                red_center = tuple(map(int, red_circle[0]))
                white_center = tuple(map(int, white_circle[0]))
                red_radius = int(red_circle[1])
                white_radius = int(white_circle[1])
                
                # Merkezler arası mesafe kontrolü
                center_dist = np.sqrt((red_center[0] - white_center[0])**2 + 
                                    (red_center[1] - white_center[1])**2)
                
                # Daireler iç içe ve merkezleri yakın mı kontrol et
                if center_dist < 30 and red_radius > white_radius:
                    # Hedefi çiz
                    cv2.circle(frame, red_center, red_radius, (0, 0, 255), 3)
                    cv2.circle(frame, white_center, white_radius, (255, 255, 255), 3)
                    cv2.circle(frame, red_center, 5, (0, 255, 0), -1)
                    
                    # Bilgileri göster
                    cv2.putText(frame, f"Merkez: {red_center}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Oran: {red_radius/white_radius:.2f}", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Debug görüntüleri (isteğe bağlı)
        debug = np.hstack([red_mask, white_mask])
        cv2.imshow('Mask Debug', debug)
        
        # Ana görüntüyü göster
        cv2.imshow('Target Takip', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_target_realtime()
