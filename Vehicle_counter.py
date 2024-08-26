import cv2
import numpy as np
from collections import deque
import time  # 新增: 用於實現冷卻時間功能

# 影片輸入與輸出的路徑
video_path = "D:/Harry/ITS/Vehiclecounter/Video/test_5.mp4" 
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4"  

# 全局變量，用於生成唯一ID
next_vehicle_id = 0

class Vehicle:
    def __init__(self, position):
        global next_vehicle_id
        self.id = next_vehicle_id  # 唯一識別ID
        next_vehicle_id += 1
        self.positions = deque(maxlen=5) # 保存最近10個位置
        self.update_position(position)
        self.counted = set()  # 用集合記錄已經被計數的區間
        self.last_count_time = {}  # 記錄每個區域的最後計數時間
        self.last_seen = time.time()

    def update_position(self, new_position):
        self.positions.append(new_position)
        self.last_seen = time.time()

    def get_average_position(self):
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))

def vehicle_count(video_path, output_path, output_mode='original'):
    
    # 定義多個偵測區間 [x1, y1, x2, y2]
    detection_zones = [
        {"coords": [250, 560, 1280, 600], "color": (125, 0, 255), "count": 0},  # 單一偵測區間
        #{"coords": [250, 560, 530, 600], "color": (255, 0, 0), "count": 0},   # 藍色區間
        #{"coords": [530, 540, 790, 590], "color": (0, 255, 102), "count": 0},  # 綠色區間
        #{"coords": [790, 530, 1050, 590], "color": (0, 255, 255), "count": 0},  # 黃色區間
        #{"coords": [980, 440, 1200, 480], "color": (0, 165, 255), "count": 0},  # 橙色區間
    ]
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return

    # 儲存影片 
    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片格式
    #fps = 30.0 # 影片輸出幀率
    #out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))  # 設定輸出檔案參數
    
    # 影片輸出設置
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 根據輸出模式決定輸出的影片
    if output_mode == "original" :
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    elif output_mode == "binary":
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)


    # 背景減法
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    vehicles = {}
    total_count = 0  # 添加總計數變量

    cooldown_time = 2  # 冷卻時間（秒）
    time_window = 1  # 1秒內認為是同一輛車  
    zone_recent_vehicles = [{} for _ in detection_zones]  # 每個區域最近檢測到的車輛ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        # 使用高斯模糊減少雜訊
        blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
        fg_mask = background_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # 形態學操作
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_vehicles = set()
        current_time = time.time()  # 獲取當前時間
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # 閾值調整
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    vehicle_id = None

                    # 新增: 查找最近的現有車輛
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
                        if (zone["coords"][0] <= avg_pos[0] <= zone["coords"][2] and 
                            zone["coords"][1] <= avg_pos[1] <= zone["coords"][3]):
                            
                            # 新增: 檢查該區域最近檢測到的車輛
                            if vehicle_id not in zone_recent_vehicles[i]:
                                # 新車輛進入區域
                                zone_recent_vehicles[i][vehicle_id] = current_time
                                if i not in vehicles[vehicle_id].counted:

                                    # 新增: 檢查冷卻時間
                                    if i not in vehicles[vehicle_id].last_count_time or \
                                    current_time - vehicles[vehicle_id].last_count_time[i] > cooldown_time:
                                        zone["count"] += 1
                                        vehicles[vehicle_id].counted.add(i)
                                        vehicles[vehicle_id].last_count_time[i] = current_time
                                        total_count += 1  # 更新總計數

                            else:
                                # 更新最後看到的時間
                                zone_recent_vehicles[i][vehicle_id] = current_time
                    
                    cv2.circle(frame, avg_pos, 5, (0, 0, 255), -1)
                    
                    # 在車輛旁邊顯示ID
                    #cv2.putText(frame, f"ID: {vehicles[vehicle_id].id}", (avg_pos[0] + 10, avg_pos[1] - 10),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)   

        # 清理舊的車輛記錄
        for i, zone_vehicles in enumerate(zone_recent_vehicles):
            zone_recent_vehicles[i] = {v_id: last_seen for v_id, last_seen in zone_vehicles.items() 
                                       if current_time - last_seen < time_window}
        
        # 移除不再出現的車輛
        vehicles = {k: v for k, v in vehicles.items () if k in current_vehicles}
        
        # 繪製所有偵測區間和計數
        for i, zone in enumerate(detection_zones):
            cv2.rectangle(frame, (zone["coords"][0], zone["coords"][1]), 
                          (zone["coords"][2], zone["coords"][3]), zone["color"], 2)
            cv2.putText(frame, f"Zone {i+1}: {zone['count']}", (zone["coords"][0], zone["coords"][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)
        
        # 左上顯示總計數
        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 根據選擇的模式來寫入影片 
        if output_mode == 'original':
            out.write(frame)
        elif output_mode == 'binary':
            out.write(thresh)

        cv2.imshow("Vehicle Counting", frame)

        out.write(frame)  # 將處理後的每一幀寫入輸出影片
        
        if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()  # 釋放 相機資源
    out.release()  # 釋放 VideoWriter 資源
    cv2.destroyAllWindows()
    
    return [zone["count"] for zone in detection_zones], total_count

# 使用示例

zone_counts, total_count = vehicle_count(video_path, output_path)
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
    
print(f"Total vehicle count: {total_count}")
