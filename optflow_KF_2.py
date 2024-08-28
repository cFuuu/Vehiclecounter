import cv2
import numpy as np

video_path = "D:/Harry/ITS/Vehiclecounter/Video/Shulin/Shulin_2.mp4" 

# 初始化參數
max_corners = 200  # 用於光流的最大角點數
quality_level = 0.01  # 角點品質的閾值
min_distance = 10  # 角點間的最小距離
block_size = 7  # 用於檢測角點的區域大小
lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化卡爾曼濾波器參數
state_size = 4  # 狀態變數數量 (x, y, dx, dy)
meas_size = 2  # 測量變數數量 (x, y)
kalman_filters = []

# 建立一個背景減法器
fgbg = cv2.createBackgroundSubtractorMOG2()

def create_kalman_filter():
    kf = cv2.KalmanFilter(state_size, meas_size)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(state_size, dtype=np.float32) * 0.05  # 調整後的過程噪聲
    kf.measurementNoiseCov = np.eye(meas_size, dtype=np.float32) * 0.3 # 調整後的測量噪聲
    kf.errorCovPost = np.eye(state_size, dtype=np.float32)
    return kf

def update_kalman_filter(kf, measurement):
    kf.correct(measurement)
    prediction = kf.predict()
    return prediction

def track_and_assign_ids(frame, corners, kalman_filters):
    new_corners = []
    new_kalman_filters = []
    for i, corner in enumerate(corners):
        x, y = corner.ravel()
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        if i < len(kalman_filters):
            prediction = update_kalman_filter(kalman_filters[i], measurement)
            x_pred, y_pred = int(prediction[0]), int(prediction[1])
            cv2.circle(frame, (x_pred, y_pred), 4, (0, 255, 0), -1)  # 預測位置顯示為綠色圓點
            cv2.putText(frame, f'ID {i}', (x_pred + 10, y_pred), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            new_kalman_filters.append(kalman_filters[i])
        else:
            kf = create_kalman_filter()
            prediction = update_kalman_filter(kf, measurement)
            new_kalman_filters.append(kf)
        new_corners.append(corner)
    
    return np.array(new_corners), new_kalman_filters  # 將new_corners轉換為NumPy陣列後返回

cap = cv2.VideoCapture(video_path)  # 更換為你的影像檔路徑

# 初始化第一幀
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(old_gray, max_corners, quality_level, min_distance, blockSize=block_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 應用背景減法
    fgmask = fgbg.apply(gray)
    
    # 計算稀疏光流
    if corners is not None and len(corners) > 0:  # 確保corners不是空的
        new_corners, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, corners, None, **lk_params)
        good_new = new_corners[status == 1]
        good_old = corners[status == 1]
        
        # 更新卡爾曼濾波器並分配ID
        good_new, kalman_filters = track_and_assign_ids(frame, good_new, kalman_filters)
        
        old_gray = gray.copy()
        corners = good_new.reshape(-1, 1, 2)  # 確保corners為NumPy陣列並重塑形狀
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()