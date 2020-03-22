import cv2
import os

file = "./test_images/test5_succeed.jpg"
file_name = os.path.splitext(file)[0]
file_ext = os.path.splitext(file)[1]

# 画像の読み込み
image = cv2.imread(file)

# 処理速度を高めるために画像をグレースケールに変換
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 評価器を読み込み
cascade = cv2.CascadeClassifier("tools/haarcascade_frontalface_alt.xml")

# 顔検出
facerect = cascade.detectMultiScale(
    image,
    scaleFactor=1.02,
    minNeighbors=3,
    minSize=(30, 30),
    maxSize = (1000,1000)
)

if 0 != len(facerect):
    BORDER_COLOR = (255, 255, 255) # 線色を白に
    for rect in facerect:
        # 顔検出した部分に枠を描画
        cv2.rectangle(
            image,
            tuple(rect[0:2]),
            tuple(rect[0:2] + rect[2:4]),
            BORDER_COLOR,
            thickness=2
        )

# 結果の画像を保存
cv2.imwrite(file_name + "_detected"  + file_ext, image)