import cv2
import numpy as np
from collections import deque
import time
from scipy.optimize import linear_sum_assignment  # 匈牙利演算法

# 影片輸入/輸出路徑
video_path = "D:/Harry/ITS/Vehiclecounter/Video/Shulin/Shulin_6.mp4" 
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4"

next_vehicle_id = 0  # 全局變量，用於生成唯一ID
class KalmanTracker:
    def __init__(self, position):
        global next_vehicle_id
        self.id = next_vehicle_id  # 唯一識別ID
        next_vehicle_id += 1

        # 初始化卡爾曼濾波器
        self.kalman = cv2.KalmanFilter(4, 2)  # 狀態向量維度4 (x, y, dx, dy)，觀測向量維度2 (x, y)
        
        # 定義狀態轉移矩陣 A
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], dtype=np.float32)
        
        # 定義觀測矩陣 H
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], dtype=np.float32)
        
        # 初始化狀態協方差矩陣 P
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # 假設過程噪聲很小
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1  # 假設觀測噪聲
        
        # 設置初始狀態
        self.kalman.statePre = np.array([[position[0]], [position[1]], [0], [0]], dtype=np.float32)
        self.kalman.statePost = np.array([[position[0]], [position[1]], [0], [0]], dtype=np.float32)

        # 初始化最近位置保存的deque
        self.positions = deque(maxlen=10)
        self.update_position(position)
        self.counted = set()  # 用集合記錄已經被計數的區間
        self.last_count_time = {}  # 記錄每個區域的最後計數時間
        self.last_seen = time.time()

    def predict(self):
        # 進行卡爾曼預測
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])

    def update_position(self, new_position):
        # 使用觀測更新卡爾曼濾波器
        self.kalman.correct(np.array([[np.float32(new_position[0])], [np.float32(new_position[1])]]))
        self.positions.append(new_position)
        self.last_seen = time.time()

    def get_average_position(self):
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))


def vehicle_count(video_path, output_path, output_mode='original'):
    detection_zones = [
        {"coords": [(9, 286),(0, 350),(180, 336),(179, 273)], "color": (100, 100, 255), "count": 0},    # 0 桃色區間(路肩)
        {"coords": [(179, 273),(180, 336),(343, 321),(315, 262)], "color": (255, 150, 0), "count": 0},  # 1 藍色區間
        {"coords": [(315, 262),(343, 321),(532, 302),(476, 249)], "color": (110, 255, 0), "count": 0},  # 2 綠色區間
        {"coords": [(476, 249),(532, 302),(650, 289),(579, 242)], "color": (0, 255, 255), "count": 0},  # 3 黃色區間
        {"coords": [(579, 242),(650, 289),(807, 272),(719, 226)], "color": (0, 165, 255), "count": 0},  # 4 橙色區間
        ]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if output_mode == "original" :
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    elif output_mode == "binary":
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)
    
    trackers = {}  # 使用卡爾曼濾波器追蹤的車輛
    total_count = 0  # 添加總計數變量
    cooldown_time = 0.5  # 冷卻時間 (秒)
    time_window = 0.4   # __秒內認為是同一輛車
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  # 背景減法
    zone_recent_vehicles = [{} for _ in detection_zones]  # 每個區域最近檢測到的車輛ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        # 使用高斯模糊減少雜訊
        blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
        fg_mask = background_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        # 形態學操作
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_vehicles = set()
        current_time = time.time()  # 獲取當前時間
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # 閾值調整
                # 計算邊界框
                x, y, w, h = cv2.boundingRect(contour)


                # 使用邊界框的中心點
                cx, cy = x + w // 2, y + h // 2  
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    vehicle_id = None

                    # 查找最近的現有車輛
                    for v_id, vehicle in vehicles.items():
                        if np.linalg.norm(np.array(vehicle.get_average_position()) - np.array((cx, cy))) < 100 and \
                            current_time - vehicle.last_seen < time_window:
                            vehicle_id = v_id
                            break

                    if vehicle_id is None:
                        vehicle_id = len(vehicles)
                        vehicles[vehicle_id] = Vehicle((cx, cy))
                    else:
                        vehicles[vehicle_id].update_position((cx, cy))

                    current_vehicles.add(vehicle_id)
                    avg_pos = vehicles[vehicle_id].get_average_position()
                    
                    for i, zone in enumerate(detection_zones):
                        if point_in_polygon(avg_pos, zone["coords"]):
                            # 檢查該區域最近檢測到的車輛
                            if vehicle_id not in zone_recent_vehicles[i]:
                                # 新車輛進入區域
                                zone_recent_vehicles[i][vehicle_id] = current_time
                                if i not in vehicles[vehicle_id].counted:
                                    # 檢查冷卻時間
                                    if i not in vehicles[vehicle_id].last_count_time or \
                                    current_time - vehicles[vehicle_id].last_count_time[i] > cooldown_time:
                                        zone["count"] += 1
                                        vehicles[vehicle_id].counted.add(i)
                                        vehicles[vehicle_id].last_count_time[i] = current_time
                                        total_count += 1  # 更新總計數
                            else:
                                # 更新最後看到的時間
                                zone_recent_vehicles[i][vehicle_id] = current_time                    

                    ##########  車輛標記 ###################################
                    # 顯示邊界框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    
                    # 顯示車輛輪廓中心
                    cv2.circle(frame, avg_pos, 5, (100, 0, 255), -1) 
                    
                    # 在車輛旁邊顯示ID
                    #cv2.putText(frame, f"ID: {vehicles[vehicle_id].id}", (avg_pos[0] + 10, avg_pos[1] - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    #######################################################

        # 清理舊的車輛記錄
        for i, zone_vehicles in enumerate(zone_recent_vehicles):
            zone_recent_vehicles[i] = { v_id: last_seen for v_id, last_seen in zone_vehicles.items() 
            if current_time - last_seen < time_window }
        
        # 移除不再出現的車輛
        vehicles = {k: v for k, v in vehicles.items () if k in current_vehicles}
        
        ##########  畫面即時顯示：計數、偵測區 #############
        
        for zone in detection_zones:
            pts = np.array(zone["coords"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, zone["color"], 2)  # 偵測區域
            cv2.putText(frame, f"Count: {zone['count']}",   # 單車道計數
                        (zone["coords"][0][0] +5, zone["coords"][0][1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)         
        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (890, 40),  # 總計數
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ##########  輸出影像模式 (原始圖、二值化)  #################
        if output_mode == 'original':
            out.write(frame)
        elif output_mode == 'binary':
            out.write(thresh)

        # 預覽視窗大小調整
        #frame = cv2.resize(frame, (1280, 720)) # 直接調整
        #frame = cv2.resize(frame, dsize=None, fx= 0.6, fy= 0.6, interpolation=None) # 比例調整
        cv2.imshow("Vehicle Counter", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()  # 釋放相機資源
    out.release()  # 釋放儲存影像資源
    cv2.destroyAllWindows()
    
    return [zone["count"] for zone in detection_zones], total_count


######  Output value (Terminal Show)  ################################
zone_counts, total_count = vehicle_count(video_path, output_path)
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
    
print(f"Total vehicle count: {total_count}")
