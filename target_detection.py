import cv2
import numpy as np

def detect_target(frame):
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=100,
        param2=40,
        minRadius=20,
        maxRadius=200
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for i in range(len(circles)):
            (x1, y1, r1) = circles[i]
            center_count = 1
            for j in range(len(circles)):
                if i == j:
                    continue
                (x2, y2, r2) = circles[j]
                center_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if center_distance < 10:
                    center_count += 1

            if center_count >= 1:
                for (x, y, r) in circles:
                    if np.sqrt((x - x1)**2 + (y - y1)**2) < 10:
                        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
                        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
                cv2.putText(output, "Hedef Tespit Edildi", (x1 - 40, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break

    return output

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_target(frame)
    cv2.imshow("Hedef Takibi", result)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
