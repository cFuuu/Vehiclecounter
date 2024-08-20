import cv2
import numpy as np

def vehicle_count(video_path, detection_zone):
    cap = cv2.VideoCapture("D:/Vehiclecounter/Video/test1.mp4")
    
    if not cap.isOpened():
        print("無法開啟影片")
        return

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    vehicle_count = 0
    tracked_vehicles = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 應用背景減除
        fg_mask = background_subtractor.apply(frame)
        
        # 應用閾值
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        # 查找輪廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # 調整此閾值以適應您的場景
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 檢查質心是否在偵測區間內
                    if detection_zone[0] <= cx <= detection_zone[2] and detection_zone[1] <= cy <= detection_zone[3]:
                        vehicle_id = hash((cx, cy))  # 使用座標的哈希值作為簡單的ID
                        if vehicle_id not in tracked_vehicles:
                            vehicle_count += 1
                            tracked_vehicles.add(vehicle_id)
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # 繪製偵測區間
        cv2.rectangle(frame, (detection_zone[0], detection_zone[1]), 
                      (detection_zone[2], detection_zone[3]), (255, 0, 0), 2)
        #cv2.rectangle(frame, (detection_zone[480],detection_zone[270]),(detection_zone[490],detection_zone[540]), (255, 0, 0), 2)
        # 顯示車輛計數
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

# 讓使用者輸入偵測區間的座標
print("請輸入偵測區間的座標 (x1 y1 x2 y2):")
detection_zone = list(map(int, input().split()))

total_vehicles = vehicle_count(video_path, detection_zone)
print(f"Total vehicles counted: {total_vehicles}")