import cv2
import datetime

# 初始化背景減法器
fgbg = cv2.createBackgroundSubtractorMOG2()

# 初始化攝影機
cap = cv2.VideoCapture("D:/Vehiclecounter/Video/test1.mp4")  
# 0 表示使用第一個攝影機，如果是其他攝影機，可以調整為 1, 2, ...

# 記錄已經偵測過的車輛
detected_cars = []

# 計算通過區間的車輛數量
car_count = 0

# 車輛識別的最小面積閾值（根據實際情況調整）
MIN_CAR_AREA = 2000

# 主循環，處理攝影機捕捉到的每一帧影像
while True:
    ret, frame = cap.read()  # 讀取一帧影像
    if not ret:
        break
    
    # 背景減法處理
    fgmask = fgbg.apply(frame)
    
    # 執行形態學操作，以便更好地提取前景物體
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # 找出前景物體的輪廓
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 當有新車輛被偵測到時，記錄當前時間並更新計數
    for contour in contours:
        # 計算輪廓的外界矩形框
        (x, y, w, h) = cv2.boundingRect(contour)
        car_area = w * h
        
        # 假設車輛的面積大於最小面積閾值才算是車輛，避免小雜物影響
        if car_area > MIN_CAR_AREA:
            car_id = (x, y, w, h)
            if car_id not in detected_cars:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                detected_cars.append(car_id)
                car_count += 1
                print(f"Vehicle detected at {current_time}")
                # 在這裡可以將車輛通過的時間及其他相關資訊寫入數據庫或文件
                cv2.putText(frame, f"Vehicle {car_count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 顯示處理後的影像
    cv2.imshow('Foreground Mask', fgmask)
    cv2.imshow('Original Frame', frame)
    
    # 按下 ESC 鍵退出迴圈
    if cv2.waitKey(100) & 0xFF == 27:
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
