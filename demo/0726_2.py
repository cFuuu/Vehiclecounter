import cv2
import numpy as np
from collections import deque

class Vehicle:
    def __init__(self, position):
        self.positions = deque(maxlen=10)  # 保存最近5個位置
        self.update_position(position)
        self.counted = False

    def update_position(self, new_position):
        self.positions.append(new_position)

    def get_average_position(self):
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))

def vehicle_count(video_path):
    detection_zone = [200, 490, 1280, 500]  # [x1, y1, x2, y2]
    
    cap = cv2.VideoCapture("D:/Vehiclecounter/Video/LaneCarVideo_1.mp4")
    if not cap.isOpened():
        print("無法開啟影片")
        return

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    vehicles = {}
    vehicle_count = 0

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
        
        for contour in contours:
            if cv2.contourArea(contour) > 500: #閾值調整
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
                    
                    if (detection_zone[0] <= avg_pos[0] <= detection_zone[2] and 
                        detection_zone[1] <= avg_pos[1] <= detection_zone[3] and 
                        not vehicles[vehicle_id].counted):
                        vehicle_count += 1
                        vehicles[vehicle_id].counted = True
                    
                    cv2.circle(frame, avg_pos, 5, (0, 255, 0), -1)
        
        # 移除不再出現的車輛
        vehicles = {k: v for k, v in vehicles.items() if k in current_vehicles}
        
        cv2.rectangle(frame, (detection_zone[0], detection_zone[1]), 
                      (detection_zone[2], detection_zone[3]), (255, 0, 0), 2)
        
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Vehicle Counting", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return vehicle_count

# 使用示例
video_path = "path_to_your_video.mp4"
total_vehicles = vehicle_count(video_path)
print(f"Total vehicles counted: {total_vehicles}")