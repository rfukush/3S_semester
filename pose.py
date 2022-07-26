import cv2
#mediapipeで手の認識
import mediapipe as mp
import time
import numpy as np
import random

#cameraからキャプチャ
cap = cv2.VideoCapture(0)

#handや姿勢推定のためのモデルを初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils 
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

#距離計算
#二点の座標が入力
def calcDistance(p0, p1):
    a1 = p1.x-p0.x
    a2 = p1.y-p0.y
    return np.sqrt(a1*a1 + a2*a2)

#角度計算
#三点の座標が入力
def calcAngle(p0, p1, p2):
    a1 = p1.x-p0.x
    a2 = p1.y-p0.y
    b1 = p2.x-p1.x
    b2 = p2.y-p1.y
    angle = np.arccos( (a1*b1 + a2*b2) / np.sqrt((a1*a1 + a2*a2)*(b1*b1 + b2*b2)) ) * 180/ np.pi
    return angle

#指の関節の角度をすべて足し算
def cancFingerAngle(p0, p1, p2, p3, p4):
    result = 0
    result += calcAngle(p0, p1, p2)
    result += calcAngle(p1, p2, p3)
    result += calcAngle(p2, p3, p4)
    return result

#ポーズの推定
#okだけは曲がり角度が他よりも小さいので、ひとつだけ別で判断した
def detectFingerPose(landmarks):
    thumbIsOpen = cancFingerAngle(landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]) < 70
    firstFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[5], landmarks[6], landmarks[7], landmarks[8]) < 100
    secondFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[9], landmarks[10], landmarks[11], landmarks[12]) < 100
    thirdFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[13], landmarks[14], landmarks[15], landmarks[16]) < 100
    fourthFingerIsOpen = cancFingerAngle(landmarks[0], landmarks[17], landmarks[18], landmarks[19], landmarks[20]) < 100
    ok_thumb = cancFingerAngle(landmarks[0], landmarks[1], landmarks[2], landmarks[3], landmarks[4]) > 50
    ok_first = cancFingerAngle(landmarks[0], landmarks[5], landmarks[6], landmarks[7], landmarks[8]) > 80

    if (not thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen):
        return "No.1"
    elif(thumbIsOpen and not firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen):
        return "Good"
    elif(thumbIsOpen and firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and fourthFingerIsOpen):
        return "I love you"
    elif(not thumbIsOpen and firstFingerIsOpen and secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen):
        return "Peace"
    elif(thumbIsOpen and not firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and fourthFingerIsOpen):
        return "Thank you"
    elif(not thumbIsOpen and not firstFingerIsOpen and not secondFingerIsOpen and not thirdFingerIsOpen and not fourthFingerIsOpen):
        return "Fist"
    elif(ok_thumb and ok_first and secondFingerIsOpen and thirdFingerIsOpen and fourthFingerIsOpen):
        return "OK"

#７つのサイン
sign_list = ["No.1", "Good", "Peace", "Thank you", "OK", "I love you", "Fist"]
#得点
score = 0
#お題の更新をするか
topic = True
#お題をクリアしたか
clear = False
#人数をコマンドラインから入力（最大で２つまでの手しか認識できない）
player = int(input("人数を選んでください : "))

while True:
    #画面をキャプチャ
    _, img = cap.read()
    #画面サイズを変更
    img = cv2.resize(img, dsize = (1000, 700))
    #BGRの画像をRGBに変換
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #姿勢、顔、手の認識をする（今回は手だけなので用いなかった）
    #results = holistic.process(imgRGB)

            
    """if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            cx, cy = int(lm.x * width), int(lm.y * height)
            #if i > 10:
                #cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                #cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) 
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, draw_spec, draw_spec) """

    #クリアしたかどうかを判定し、クリアしたらお題を変え、得点を1プラスする
    if clear:
        topic = True
        score += 1
        clear = False

    #お題を変える
    if topic:
        x = random.sample(sign_list, 1) #お題はリストからランダムで決める
        time_sta = time.time() #お題が変わってからの時間を測る。測り始め。
        topic = False

    #お題とスコアの表示
    cv2.putText(img, x[0], (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.putText(img, f"score: {score}", (650, 50), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 5, cv2.LINE_AA)

    #手の認識をする
    results = hands.process(imgRGB)

    #手を認識できたとき
    if results.multi_hand_landmarks:
        #flagは2人のときに両方正解かを判断するためのもの
        flag = True
        #countで人数を判断
        count = 0
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            """ for i, lm in enumerate(hand_landmarks.landmark):
                height, width, channel = img.shape
                #cx, cy = int(lm.x * width), int(lm.y * height)
                #cv2.putText(img, str(i+1), (cx+10, cy+10), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 5, cv2.LINE_AA)
                #cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) """
            #自分の現在のポーズを表示。二人の場合、位置をずらして表示
            img = cv2.putText(img, text = detectFingerPose(hand_landmarks.landmark), org = (10 + i * 500,600), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3, color = (255,0,0), thickness = 7)
            #お題と違うとflagをFalseとする
            if detectFingerPose(hand_landmarks.landmark) != x[0]:
                flag = False
            #手の関節に合わせて線や点を表示する
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            count += 1
        #人数に合わせてクリアかどうか
        if flag and (count >= player):
            clear = True

    cv2.imshow("Image", img)

    #制限時間によってゲームオーバーか決める
    time_end = time.time()
    if time_end - time_sta > 5:
        print("Game Over")
        break
    if score >= 10:
        print("Game Clearrrrrrrrrrrrrrr!!!!!!")
        break
    k = cv2.waitKey(1)
    if k == 13:
        break