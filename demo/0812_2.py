import cv2
import numpy as np
from collections import deque

class Vehicle:
    def __init__(self, position):
        self.positions = deque(maxlen=3)  # 保存最近3個位置
        self.update_position(position)
        self.counted = set()  # 用集合記錄已經被計數的區間

    def update_position(self, new_position):
        self.positions.append(new_position)

    def get_average_position(self):
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))

def vehicle_count(video_path, output_path, window_scale=1.0):
    # 定義多個偵測區間 [x1, y1, x2, y2]
    detection_zones = [
        {"coords": [250, 590, 530, 600], "color": (255, 0, 0), "count": 0},   # 藍色區間
        {"coords": [525, 540, 800, 570], "color": (0, 255, 102), "count": 0},  # 綠色區間
        {"coords": [800, 525, 1050, 560], "color": (0, 255, 255), "count": 0},  # 黃色區間
        {"coords": [1000, 450, 1200, 480], "color": (0, 165, 255), "count": 0},  # 橙色區間
        # 可以繼續添加更多區間
    ]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return

    # 影片輸出設置
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    vehicles = {}
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        blurred = cv2.GaussianBlur(frame, (5, 5), 0) 
        fg_mask = background_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_vehicles = set()
        
        for contour in contours:
            if cv2.contourArea(contour) > 1500:  # 閾值調整
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    vehicle_id = hash((cx, cy))
                    
                    if vehicle_id not in vehicles:
                        vehicles[vehicle_id] = Vehicle((cx, cy))
                    else:
                        vehicles[vehicle_id].update_position((cx, cy))
                    
                    current_vehicles.add(vehicle_id)
                    
                    avg_pos = vehicles[vehicle_id].get_average_position()

                    for i, zone in enumerate(detection_zones):
                        if (zone["coords"][0] <= avg_pos[0] <= zone["coords"][2] and 
                            zone["coords"][1] <= avg_pos[1] <= zone["coords"][3] and 
                            i not in vehicles[vehicle_id].counted):
                            zone["count"] += 1
                            vehicles[vehicle_id].counted.add(i)
                            
                            total_count += 1
                    
                    # 在影像上標記輪廓中心
                    cv2.circle(frame, avg_pos, 5, (0, 0, 255), -1)
        
        vehicles = {k: v for k, v in vehicles.items() if k in current_vehicles}
        
        for i, zone in enumerate(detection_zones):
            cv2.rectangle(frame, (zone["coords"][0], zone["coords"][1]), 
                          (zone["coords"][2], zone["coords"][3]), zone["color"], 2)
            cv2.putText(frame, f"Zone {i+1}: {zone['count']}", (zone["coords"][0], zone["coords"][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)
        
        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 調整影像大小
        resized_frame = cv2.resize(frame, (int(frame_width * window_scale), int(frame_height * window_scale)))
        resized_thresh = cv2.resize(thresh, (int(frame_width * window_scale), int(frame_height * window_scale)))
        
        # 顯示原始影像與二值化影像
        #cv2.imshow("Original Frame", resized_frame)
        #cv2.imshow("Binary Frame", resized_thresh)
        cv2.imshow("Vehicle counting", resized_frame)
        # 保存原始影像影片
        out.write(frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    out.release()  # 釋放影片寫入資源
    cv2.destroyAllWindows()
    
    return [zone["count"] for zone in detection_zones], total_count

# 使用示例
video_path = "D:/Harry/ITS/Vehiclecounter/Video/test_3.mp4" # 影片輸入路徑
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4"  # 輸出影片路徑
zone_counts = vehicle_count(video_path, output_path, window_scale=0.5)
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
print(f"Total vehicle count: {total_count}")
