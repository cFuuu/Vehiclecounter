import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import time

# 影片輸入與輸出的路徑
video_path = "D:/Harry/ITS/Vehiclecounter/Video/Shulin/Shulin_2.mp4" 
output_path = "D:/Harry/ITS/Vehiclecounter/Outputvideo/outputvideo.mp4" 


class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        self.kf.x[:4] = bbox.reshape((4, 1))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(bbox.reshape((4, 1)))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].reshape((1,4)))
        return self.history[-1]

    def get_state(self):
        return self.kf.x[:4].reshape((1,4))[0]

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    if len(trackers)==0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    matched_indices = linear_sum_assignment(-iou_matrix)
    matched_indices = np.asarray(matched_indices).T
    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)

class Sort(object):
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0

    def update(self, dets):
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

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

    # 初始化 SORT 追蹤器
    mot_tracker = Sort()

    # 定義偵測區間
    detection_zones = [
        {
            "coords": [(250, 520), (530, 540), (530, 560), (250, 540)],  # 藍色區間
            "color": (255, 0, 0),
            "count": 0
        },
        # ... 其他區間 ...
    ]

    zone_recent_vehicles = [{} for _ in detection_zones]
    cooldown_time = 2
    time_window = 1
    total_count = 0

    background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        fg_mask = background_subtractor.apply(blurred)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            if cv2.contourArea(contour) > 1500:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append([x, y, x+w, y+h])

        trackers = mot_tracker.update(np.array(detections))

        for d in trackers:
            d = d.astype(np.int32)
            vehicle_id = d[4]
            cx, cy = (d[0] + d[2]) // 2, (d[1] + d[3]) // 2

            for i, zone in enumerate(detection_zones):
                if point_in_polygon((cx, cy), zone["coords"]):
                    if vehicle_id not in zone_recent_vehicles[i]:
                        zone_recent_vehicles[i][vehicle_id] = current_time
                        if i not in vehicles[vehicle_id].counted:
                            if i not in vehicles[vehicle_id].last_count_time or \
                            current_time - vehicles[vehicle_id].last_count_time[i] > cooldown_time:
                                zone["count"] += 1
                                vehicles[vehicle_id].counted.add(i)
                                vehicles[vehicle_id].last_count_time[i] = current_time
                                total_count += 1
                    else:
                        zone_recent_vehicles[i][vehicle_id] = current_time

            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {vehicle_id}", (d[0], d[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 繪製偵測區間和計數
        for zone in detection_zones:
            pts = np.array(zone["coords"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, zone["color"], 2)
            cv2.putText(frame, f"Count: {zone['count']}", 
                        (zone["coords"][0][0], zone["coords"][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone["color"], 2)

        cv2.putText(frame, f"Total Vehicle Count: {total_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if output_mode == 'original':
            out.write(frame)
        elif output_mode == 'binary':
            out.write(thresh)

        cv2.imshow("Vehicle Counting", cv2.resize(frame, (1280, 720)))
        
        if cv2.waitKey(30) & 0xFF == 27:
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