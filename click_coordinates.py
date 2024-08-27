import cv2

# 初始化點擊次數
click_count = 0

# 回調函數，當滑鼠事件發生時被調用
def click_event(event, x, y, flags, param):
    global img, click_count
    if event == cv2.EVENT_LBUTTONDOWN:
        # 更新點擊次數
        click_count += 1
        
        # 在點擊的位置繪製一個圓
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # 清除右上角的座標顯示（填充一個黑色的矩形來覆蓋之前的文字）
        img[0:50, 0:170] = (0, 0, 0)

        # 在右上角顯示座標
        coord_text = f"({x}, {y})"
        cv2.putText(img, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 顯示點擊的順序號碼
        coord_text = f"{click_count}"
        cv2.putText(img, coord_text, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

        # 點擊順序和座標
        print(f"No. {click_count}: ({x}, {y}),")

        # 更新顯示的圖片
        cv2.imshow('image', img)

# 載入圖片
img = cv2.imread('D:/Harry/ITS/Vehiclecounter/Image/Shulin/0.jpg')
# 顯示圖片
cv2.imshow('image', img)
# 註冊滑鼠回調函數
cv2.setMouseCallback('image', click_event)

# 保持窗口打開直到按下任意鍵
while True:
    if cv2.waitKey(30) & 0xFF == 27:  # 按ESC退出
        break

cv2.destroyAllWindows()

