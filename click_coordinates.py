import cv2

# 查看影像任意點位的座標

# 回調函數，當滑鼠事件發生時被調用
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 在點擊的位置繪製一個圓
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        # 顯示座標
        print(f"座標: ({x}, {y})")
        # 更新顯示的圖片
        cv2.imshow('image', img)

# 載入圖片
img = cv2.imread('D:\\Harry\\ITS\\Vehiclecounter\\Image\\20240807_1\\0.jpg')
# 顯示圖片
cv2.imshow('image', img)
# 註冊滑鼠回調函數
cv2.setMouseCallback('image', click_event)

# 保持窗口打開直到按下任意鍵
cv2.waitKey(0)
cv2.destroyAllWindows()

