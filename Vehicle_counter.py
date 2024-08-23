import cv2
import numpy as np
from collections import deque

# 影片輸入路徑
video_path = "D:/Harry/ITS/Vehiclecounter/Video/test_5.mp4" 

# 輸出影片路徑 
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4"  

def get_vehicle_center(contour, method='both'):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx_contour = int(M["m10"] / M["m00"])
        cy_contour = int(M["m01"] / M["m00"])
    else:
        cx_contour, cy_contour = 0, 0
    
    x, y, w, h = cv2.boundingRect(contour)
    cx_bbox, cy_bbox = int(x + w/2), int(y + h/2)
    
    if method == 'both':
        return (int((cx_contour + cx_bbox) / 2), int((cy_contour + cy_bbox) / 2))
    elif method == 'contour':
        return (cx_contour, cy_contour)
    else:  # bounding box
        return (cx_bbox, cy_bbox)



class Vehicle:
    def __init__(self, position):
        self.positions = deque(maxlen=3)  # 保存最近3個位置
        self.update_position(position)
        self.counted = set()  # 用集合記錄已經被計數的區間
        self.direction = None

    def update_position(self, new_position):
         if self.positions:
            old_pos = self.positions[-1]
            dx = new_position[0] - old_pos[0]
            dy = new_position[1] - old_pos[1]
            distance = (dx**2 + dy**2)**0.5
            if distance > 5:  # 避免小的抖動影響方向判斷
                self.direction = (dx/distance, dy/distance)
            self.positions.append(new_position)
        

    def get_average_position(self):
        return (int(sum(x for x, y in self.positions) / len(self.positions)),
                int(sum(y for x, y in self.positions) / len(self.positions)))

    def adjust_position(self, frame_height, frame_width):
        if not self.direction:
            return self.get_average_position()

        avg_pos = self.get_average_position()
        # 根據方向調整位置
        adjustment = 50  # 調整的像素數，可以根據需要修改
        new_x = avg_pos[0] + int(self.direction[0] * adjustment)
        new_y = avg_pos[1] + int(self.direction[1] * adjustment)

        # 確保調整後的位置在畫面內
        new_x = max(0, min(new_x, frame_width - 1))
        new_y = max(0, min(new_y, frame_height - 1))

        return (new_x, new_y)


def vehicle_count(video_path, output_path, output_mode='original', center_method='both'):  # 新增 output_path 參數 
    # 定義多個偵測區間 [x1, y1, x2, y2]
    detection_zones = [
        {"coords": [250, 520, 530, 540], "color": (255, 0, 0), "count": 0},   # 藍色區間
        {"coords": [525, 540, 800, 570], "color": (0, 255, 102), "count": 0},  # 綠色區間
        {"coords": [800, 525, 1050, 560], "color": (0, 255, 255), "count": 0},  # 黃色區間
        {"coords": [1000, 450, 1200, 480], "color": (0, 165, 255), "count": 0},  # 橙色區間
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
    if output_mode == 'original':
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    elif output_mode == 'binary':
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=False)

    # 背景減法
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)  
    
    vehicles = {} 
    total_count = 0  # 添加總計數變量

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
        
        ### 修改部分成"邊界框中心"定義 ###

        for contour in contours:
            if cv2.contourArea(contour) > 2500:  # 閥值調整
                cx, cy = get_vehicle_center(contour, method=center_method)
                vehicle_id = hash((cx, cy))
                
                if vehicle_id not in vehicles:
                    vehicles[vehicle_id] = Vehicle((cx, cy))
                else:
                    vehicles[vehicle_id].update_position((cx, cy))
                
                current_vehicles.add(vehicle_id)
                
                adjusted_pos = vehicles[vehicle_id].adjust_position(frame_height, frame_width)
                
                for i, zone in enumerate(detection_zones):
                    if (zone["coords"][0] <= adjusted_pos[0] <= zone["coords"][2] and 
                        zone["coords"][1] <= adjusted_pos[1] <= zone["coords"][3] and 
                        i not in vehicles[vehicle_id].counted):
                        zone["count"] += 1
                        vehicles[vehicle_id].counted.add(i)
                        total_count += 1
                
                # 繪製原始位置和調整後的位置
                avg_pos = vehicles[vehicle_id].get_average_position()
                cv2.circle(frame, avg_pos, 5, (0, 255, 0), -1)
                cv2.circle(frame, adjusted_pos, 5, (0, 0, 255), -1)
                if vehicles[vehicle_id].direction:
                    end_point = (int(adjusted_pos[0] + vehicles[vehicle_id].direction[0]*30),
                                 int(adjusted_pos[1] + vehicles[vehicle_id].direction[1]*30))
                    cv2.arrowedLine(frame, adjusted_pos, end_point, (255, 0, 0), 2)

                # 繪製邊界矩形
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        
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

zone_counts, total_count = vehicle_count(video_path, output_path, center_method='both')
for i, count in enumerate(zone_counts):
    print(f"Zone {i+1} vehicle count: {count}")
    
print(f"Total vehicle count: {total_count}")
