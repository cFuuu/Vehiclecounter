import cv2
import numpy as np
from collections import deque
import time

# 影片輸入與輸出的路徑
video_path = "D:/Harry/ITS/Vehiclecounter/Video/test1/test_5.mp4" 
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4"  

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if min(p1y, p2y) < y <= max(p1y, p2y):
            if x <= max(p1x, p2x):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class Vehicle:
    def __init__(self, position):
        self.positions = deque(maxlen=10)
        self.update_position(position)
        self.counted = set()
        self.last_count_time = {}
        self.last_seen = time.time()

    def update_position(self, new_position):
        self.positions.append(new_position)
        self.last_seen = time.time()

    def get_average_position(self):
        if not self.positions:
            return None
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))

def vehicle_count(video_path, output_path, output_mode='original'):
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

    detection_zones = [
        {
            "coords": [(250, 520), (530, 540), (530, 560), (250, 540)],
            "color": (255, 0, 0),
            "count": 0
        },
        # ... [其他區域定義] ...
    ]

    zone_recent_vehicles = [{} for _ in detection_zones]
    cooldown_time = 2
    time_window = 1

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    vehicles = {}
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # 影像處理步驟
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = background_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_vehicles = set()

        for contour in contours:
            if cv2.contourArea(contour) > 2500:
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

                    if avg_pos:  # 確保 avg_pos 不是 None
                        for i, zone in enumerate(detection_zones):
                            if point_in_polygon(avg_pos, zone["coords"]):
                                if vehicle_id not in zone_recent_vehicles[i]:  # 檢查該區域最近檢測到的車輛
                                    zone_recent_vehicles[i][vehicle_id] = current_time  # 新車輛進入區域
                                    if i not in vehicles[vehicle_id].counted:
                                        if i not in vehicles[vehicle_id].last_count_time or \
                                           current_time - vehicles[vehicle_id].last_count_time[i] > cooldown_time:
                                            zone["count"] += 1
                                            vehicles[vehicle_id].counted.add(i)
                                            vehicles[vehicle_id].last_count_time[i] = current_time
                                            total_count += 1
                                else:
                                    zone_recent_vehicles[i][vehicle_id] = current_time

                        cv2.circle(frame, avg_pos, 5, (0, 0, 255), -1)
                        cv2.putText(frame, f"ID: {vehicle_id}", (avg_pos[0] + 10, avg_pos[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 清理舊的車輛記錄
        vehicles = {k: v for k, v in vehicles.items() if k in current_vehicles}

        # 繪製所有偵測區間和計數
        for zone in detection_zones:
            pts = np.array(zone["coords"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, zone["color"], 2)
            cv2.putText(frame, f"Count: {zone['count']}", 
                        (zone["coords"][0][0], zone["coords"][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)

        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("Vehicle Counting", frame)

        if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return [zone["count"] for zone in detection_zones], total_count

# 使用示例

zone_counts, total_count = vehicle_count(video_path, output_path)
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
print(f"Total vehicle count: {total_count}")