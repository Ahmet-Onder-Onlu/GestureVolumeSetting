import cv2
import mediapipe as mp
import pyautogui

# Webcam'i başlat
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Görüntüyü yatay olarak çevir
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR'yi RGB'ye dönüştür
    output = hand_detector.process(rgb_img)  # RGB görüntüyü kullanarak işlem yap
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(img, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark
            for id, lm in enumerate(landmarks):
                if id == 8:  # İşaret parmağı ucu
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 15, (255, 255, 255), -1)
                    cv2.putText(img, f'{int(lm.x * w)}, {int(lm.y * h)}', (cx, cy),
                                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
                    
                    # Koordinatları ekrana uygun şekilde dönüştür
                    index_x = screen_width * lm.x
                    index_y = screen_height * lm.y

                    pyautogui.moveTo(index_x, index_y)
    
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' tuşuna basıldığında çıkış yap
        break

cap.release()
cv2.destroyAllWindows()
