import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def count_fingers(contours, drawing):
    if len(contours) == 0:
        return 0
    
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    
    if defects is None:
        return 0
    
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(max_contour[s][0])
        end = tuple(max_contour[e][0])
        far = tuple(max_contour[f][0])
        
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        
        angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        
        if angle <= 90:
            count += 1
            cv2.circle(drawing, far, 3, [255,0,0], -1)
    
    return count + 1

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı.")
        break
    
    frame = cv2.flip(frame, 1)
    roi = frame[100:400, 100:400]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5,5), 100)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    finger_count = count_fingers(contours, roi)
    
    cv2.rectangle(frame, (100,100), (400,400), (0,255,0), 2)
    
    cv2.putText(frame, f"Parmak Sayisi: {finger_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("El Hareketi Algilama", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 

print("Programdan çıkıldı.")