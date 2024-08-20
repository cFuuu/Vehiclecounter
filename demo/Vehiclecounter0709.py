import cv2
import datetime

# 創建背景減除器
backSub = cv2.createBackgroundSubtractorMOG2()

# 打開影片文件或攝影機
cap = cv2.VideoCapture("D:/Vehiclecounter/Video/test1.mp4")

# 設置興趣區域（根據實際情況調整）
roi = (480, 270, 100, 300)  # (x, y, width, height)

# 初始化車輛計數和時間記錄
vehicle_count = 0
vehicle_times = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 應用背景減除
    fgMask = backSub.apply(frame)

    # 設定興趣區域
    x, y, w, h = roi
    fgMask = fgMask[y:y+h, x:x+w]
    frame_roi = frame[y:y+h, x:x+w]

    # 尋找輪廓
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 過濾小輪廓，只保留大於某個面積的輪廓
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # 面積閾值根據需要調整
            x2, y2, w2, h2 = cv2.boundingRect(contour)
            cv2.rectangle(frame_roi, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            
            # 紀錄車輛經過時間
            current_time = datetime.datetime.now()
            vehicle_times.append(current_time)
            vehicle_count += 1
            print(f"Vehicle detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 顯示結果
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)
    #cv2.imshow('Frame', cv2.resize(framestack,(960,540)))

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 輸出結果
print(f"Total vehicle count: {vehicle_count}")
for time in vehicle_times:
    print(f"Vehicle passed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
