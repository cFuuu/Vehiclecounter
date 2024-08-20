import cv2
import numpy as np

def vehicle_count(video_path):
    #cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture("D:/Vehiclecounter/Video/test1.mp4")
    
    # 獲取影片的寬度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 設定偵測線的位置 (這裡設定在畫面中間)
    line_position = width // 2
    
    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    vehicle_count = 0
    previous_centroids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = background_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_centroids = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # 調整此閾值以適應您的場景
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    current_centroids.append((cx, cy))
                    
                    # 檢查是否穿過偵測線
                    if any(prev_y > line_position and cy <= line_position for prev_x, prev_y in previous_centroids):
                        vehicle_count += 1
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        
        # 更新previous_centroids
        previous_centroids = current_centroids
        
        # 繪製偵測線
        cv2.line(frame, (0, line_position), (width, line_position), (255, 0, 0), 2)
        
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
total_vehicles = vehicle_count(video_path)
print(f"Total vehicles counted: {total_vehicles}")