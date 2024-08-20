import cv2
import datetime
import numpy as np

# 創建背景減除器
backSub = cv2.createBackgroundSubtractorKNN()

# 打開視頻文件或攝像機
cap = cv2.VideoCapture("D:/Vehiclecounter/Video/test1.mp4")

# 設置興趣區域（根據實際情況調整）
roi = (100, 200, 500, 300)  # (x, y, width, height)

# 初始化車輛計數和時間記錄
vehicle_count = 0
vehicle_times = []
tracked_objects = []

# Kalman Filter initialization
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        return self.kf.predict()

    def correct(self, coordX, coordY):
        measurement = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        return self.kf.correct(measurement)

# 追蹤物體
class TrackedObject:
    def __init__(self, id, x, y):
        self.id = id
        self.kf = KalmanFilter()
        self.predicted = self.kf.correct(x, y)
        self.age = 0
        self.last_seen = datetime.datetime.now()

    def update(self, x, y):
        self.predicted = self.kf.correct(x, y)
        self.age += 1
        self.last_seen = datetime.datetime.now()

# 設置車輛追蹤器
next_id = 0
min_dist = 50  # 追蹤最小距離閾值

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

    # 過濾掉微小移動
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)

    # 尋找輪廓
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 更新追蹤物體
    new_objects = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x2, y2, w2, h2 = cv2.boundingRect(contour)
            cx = x2 + w2 // 2
            cy = y2 + h2 // 2
            found = False
            for obj in tracked_objects:
                dist = np.linalg.norm(np.array([cx, cy]) - obj.predicted[:2])
                if dist < min_dist:
                    obj.update(cx, cy)
                    found = True
                    break
            if not found:
                new_obj = TrackedObject(next_id, cx, cy)
                new_objects.append(new_obj)
                next_id += 1
                current_time = datetime.datetime.now()
                vehicle_times.append(current_time)
                vehicle_count += 1
                print(f"Vehicle detected at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

    tracked_objects.extend(new_objects)

    # 清理過時的追蹤物體
    tracked_objects = [obj for obj in tracked_objects if (datetime.datetime.now() - obj.last_seen).seconds < 2]

    # 繪製追蹤結果
    for obj in tracked_objects:
        cv2.circle(frame_roi, (int(obj.predicted[0]), int(obj.predicted[1])), 5, (0, 255, 0), -1)
        cv2.putText(frame_roi, f"ID: {obj.id}", (int(obj.predicted[0]), int(obj.predicted[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 顯示結果
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 輸出結果
print(f"Total vehicle count: {vehicle_count}")
for time in vehicle_times:
    print(f"Vehicle passed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
