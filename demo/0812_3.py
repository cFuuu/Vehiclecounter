import cv2
import numpy as np
from collections import deque

class Vehicle:
    def __init__(self, position):
        self.positions = deque(maxlen=3)
        self.update_position(position)
        self.counted = set()
        
        # 卡爾曼濾波器初始化
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03

        self.kalman.statePre = np.array([[position[0]],
                                         [position[1]],
                                         [0],
                                         [0]], np.float32)

    def update_position(self, new_position):
        self.positions.append(new_position)
        
        # 確保卡爾曼濾波器存在
        if hasattr(self, 'kalman'):
            measurement = np.array([[np.float32(new_position[0])],
                                    [np.float32(new_position[1])]])
            self.kalman.correct(measurement)
            self.kalman.predict()

    def get_average_position(self):
        if hasattr(self, 'kalman'):
            predicted = self.kalman.statePre
            return int(predicted[0]), int(predicted[1])
        else:
            # 如果卡爾曼濾波器不存在，回退到普通平均值
            return int(sum(x for x, y in self.positions) / len(self.positions)),


def vehicle_count(video_path):
    detection_zones = [
        {"coords": [250, 590, 530, 600], "color": (255, 0, 0), "count": 0},
        {"coords": [525, 540, 800, 570], "color": (0, 255, 102), "count": 0},
        {"coords": [800, 525, 1050, 560], "color": (0, 255, 255), "count": 0},
        {"coords": [1000, 450, 1200, 480], "color": (0, 165, 255), "count": 0},
    ]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("無法開啟影片")
        return

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
            if cv2.contourArea(contour) > 5000:
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
                    
                    cv2.circle(frame, avg_pos, 5, (0, 0, 255), -1)
        
        vehicles = {k: v for k, v in vehicles.items() if k in current_vehicles}
        
        for i, zone in enumerate(detection_zones):
            cv2.rectangle(frame, (zone["coords"][0], zone["coords"][1]), 
                          (zone["coords"][2], zone["coords"][3]), zone["color"], 2)
            cv2.putText(frame, f"Zone {i+1}: {zone['count']}", (zone["coords"][0], zone["coords"][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)
        
        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Vehicle Counting", frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
    return [zone["count"] for zone in detection_zones], total_count

# 使用示例
video_path = "D:/Harry/ITS/Vehiclecounter/Video/test_5.mp4"
zone_counts = vehicle_count(video_path)
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
    print(f"Total vehicle count: {total_count}")
