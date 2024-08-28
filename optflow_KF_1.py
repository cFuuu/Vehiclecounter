import cv2
import numpy as np

video_path = "D:/Harry/ITS/Vehiclecounter/Video/Shulin/Shulin_2.mp4" 

# 定義卡爾曼濾波器的初始化參數
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)  # 4個狀態變量，2個測量變量
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0]], np.float32)  # 測量矩陣
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]], np.float32)  # 狀態轉移矩陣
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32) * 0.03  # 進程噪聲協方差
    return kalman

# 設定參數
cap = cv2.VideoCapture(video_path)  # 輸入影片
vehicle_ids = {}  # 儲存車輛ID
vehicle_count = 0  # 車輛計數
kalman_filters = []  # 儲存每輛車的卡爾曼濾波器

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 使用背景減法
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 轉換為灰階
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)  # 二值化

    # 偵測輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_vehicles = []  # 儲存偵測到的車輛位置

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 設定最小輪廓面積
            (x, y, w, h) = cv2.boundingRect(contour)
            detected_vehicles.append((x + w // 2, y + h // 2))  # 儲存中心點

    # 追蹤每輛車輛
    for vehicle in detected_vehicles:
        x, y = vehicle

        # 如果沒有現有的卡爾曼濾波器，則創建一個新的
        if vehicle_count not in vehicle_ids:
            kalman = initialize_kalman()
            kalman.statePre = np.array([[x], [y], [0], [0]], np.float32)  # 初始化狀態
            kalman_filters.append(kalman)  # 儲存卡爾曼濾波器
            vehicle_ids[vehicle_count] = (x, y)  # 註冊ID
            vehicle_count += 1
        else:
            # 更新卡爾曼濾波器
            kalman_filters[vehicle_count].correct(np.array([[np.float32(x)], [np.float32(y)]]))  # 更新測量
            prediction = kalman_filters[vehicle_count].predict()  # 預測下一位置
            cv2.circle(frame, (int(prediction[0]), int(prediction[1])), 5, (255, 0, 0), -1)  # 繪製預測位置

        # 在影像上標註車輛ID
        cv2.putText(frame, str(vehicle_count), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 顯示影像
    cv2.imshow('Vehicle Tracking', frame)

    if cv2.waitKey(30) & 0xFF == 27:  # 按Esc鍵退出
        break

cap.release()
cv2.destroyAllWindows()
