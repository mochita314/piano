# coding: UTF-8

import cv2
import numpy as np
from PIL import Image
import wave
from scipy.io import wavfile
from scipy.io.wavfile import read
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt

#配列の中で、それぞれの音に対応する要素が「１」だったらその音がならない仕組みになっているので、まず0で初期化
normal_doremi = [0,0,0,0,0,0,0,0,0,0,0,0,0]
high_doremi = [0,0,0,0,0,0,0,0,0,0,0,0,0]

#青色が検出された座標に応じて関数が返した数値で条件分岐して音を指定しているので、その数値の配列
num_list = [100,101,102,103,104,105,106,107,108,109,110,111,112,0,1,2,3,4,5,6,7,8,9,10,11]

#何番の曲を選んだかを示す数字、まったく関係のない10で初期化しておく。
check = 10

#最後に弾いた音が何かを示す変数、nullで初期化
Sound = "null"

#それぞれの音が書き込まれているファイルのリスト
file_list = ["C4.wav","C#4.wav","D4.wav","D#4.wav","E4.wav","F.wav","F#4.wav","G4.wav","G#4.wav","A4.wav","A#4.wav","B4.wav",\
                "C5.wav","C#5.wav","D5.wav","D#5.wav","E5.wav","F5.wav","F#5.wav","G5.wav","G#5.wav","A5.wav","A#5.wav","B5.wav","C6.wav"]

#音のリスト
doremi_list = ["C4","C#4","D4","D#4","E4","F","F#4","G4","G#4","A4","A#4","B4",\
                "C5","C#5","D5","D#5","E5","F5","F#5","G5","G#5","A5","A#5","B5","C6"]

#1曲目「きらきら星」の正解の音のリスト
Answer_list = ["C4","C4","G4","G4","A4","A4","G4","F","F","E4",\
                "E4","D4","D4","C4","G4","G4","F","F","E4","E4",\
                "D4","G4","G4","F","F","E4","E4","D4","C4","C4",\
                "G4","G4","A4","A4","G4","F","F","E4","E4","D4",\
                "D4","C4"]
#森のくまさん
Answer_list2 = ["G4","F#4","G4","E4","E4","D#4","E4","C4","E4",\
                "D4","C4","D4","G4","A4","G4","E4","G4","A4",\
                "B4","C5","G4","E4","C4","A4","A4","B4","A4",\
                "G4","F","E4","D4","C4"]
#愛の挨拶
Answer_list3 = ["C#5","E4","C#5","B4","A4","G#4","A4","D5","D5","D5",\
                "E4","C#5","F#4","C#5","B4","A4","G#4","A4","B4","B4",\
                "B4","C5","C#5","E4","C#5","B4","A4","G#4","A4","F#5",\
                "F#5","F#5","E5","D5","C#5","B4","A4","F#4","G#4","A4"]
#世界に一つだけの花
Answer_list4 = ["F","F","F","C5","C5","A#4","A4","G4","A4","A#4",\
                "A4","G4","F","F","F","A4","A4","G4","D4",\
                "F","F","F","G4","A4","G4","F","F","F","F",\
                "C5","C5","A#4","A4","G4","A4","A#4","A4","G4",\
                "F","F","A4","A4","G4","F","F","E4","F","F"]
#きみはロックを聴かない
Answer_list5 = ["B4","A4","G4","G4","E4","G4","E4","G4","A4","A4",\
                "D4","B4","C5","B4","A4","G4","G4","E4","G4","E4",\
                "G4","A4","A4","D4","B4","C5","B4","C5","B4","A4",\
                "G4","A4","G4","E4","G4","E4","G4","A4","A4","D4",\
                "B4","C5","B4","D5","G4","E4","F#4","G4","G4","G4",\
                "A4","G4","G4","G4","A4","G4","C5","B4","A4","G4","G4","F#4","E4","F#4","G4"]
#Pretender
Answer_list6 = ["G5","G#5","D#5","F5","G#5","D#5","F5","G#5","A#5","G#5",\
                "A#5","G#5","A#5","C6","G#5","D#5","A#5","G#5","A#5","G#5",\
                "A#5","C6","A#5","F#5","G5","F#5","G5","F#5","G5","A5",\
                "G5","D5","E5","G#4","G#4","A#5","F5","D#5","D#5","D#5",\
                "D#5","D#5","F5","A5","D#5","C6","G#5","G#5","D#5","D#5",\
                "C#5","C5","D#5","C6","A5","D#5","D#5","C#5","C5","G5",\
                "G#5","D#5","F5","G#5","D#5","F5","G#5","A#5","G#5","A#5",\
                "G#5","A#5","C6","G#5","D#5","A#5","G#5","A#5","G#5","A#5",\
                "C6","A#5","F#5","G5","F#5","G5","F#5","G5","A5","G5",\
                "D5","E5","G#4","G#4","A#5","F5","D#5","D#5","D#5","D#5",\
                "D#5","F5","G#5","A#5","C6","G#5","G#5","D#5","D#5","C#5",\
                "C5","C#5","D#5","D#5","D#5","D#5","G#5","G5","G#5"]
#実験用として高音のド２回だけ
Answer_list7 = ["C6","C6"]

fit = 0 #正しく弾いた音の数
unfit = 0 #間違えた音の数
sum = 0 #弾いた音の数（fit+unfit)
before_playing = True #ゲームを始める前か否かを示す
before_checking_the_result = True #答えをチェックする前か否かを示す

# 0 <= h <= 179 (色相)　OpenCVではmax=179なのでR:0(180),G:60,B:120となる
# 0 <= s <= 255 (彩度)　黒や白の値が抽出されるときはこの閾値を大きくする
# 0 <= v <= 255 (明度)　これが大きいと明るく，小さいと暗い
# ここでは青色を抽出するので120±20を閾値とした

LOW_COLOR_BLUE = np.array([100, 75, 75])
HIGH_COLOR_BLUE = np.array([140, 255, 255])

# 抽出する青色の塊のしきい値
AREA_RATIO_THRESHOLD = 0.005

def message():

    if before_playing == True: #ゲーム前なら、ゲームを始めるにはAを押してねとメッセージを表示
        cv2.putText(frame,'Press A key to start the game',(50,110),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),thickness = 2)
    #選んだ番号に応じて曲名表示
    if check==5:
        cv2.putText(frame,'kimi wa rokku o kikanai',(200,400),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),thickness = 2)

def find_specific_color(frame,AREA_RATIO_THRESHOLD,LOW_COLOR_BLUE,HIGH_COLOR_BLUE):
    """
    指定した範囲の色の物体の座標を取得する関数
    frame: 画像
    AREA_RATIO_THRESHOLD: area_ratio未満の塊は無視する
    LOW_COLOR_BLUE: 抽出する青色の下限(h,s,v)
    HIGH_COLOR_BLUE: 抽出する青色の上限(h,s,v)
    """
    # 高さ，幅，チャンネル数
    h,w,c = frame.shape

    # hsv色空間に変換
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    # 色を抽出する
    ex_img = cv2.inRange(hsv,LOW_COLOR_BLUE,HIGH_COLOR_BLUE)

    # 輪郭抽出
    _,contours,hierarchy = cv2.findContours(ex_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    # 面積を計算
    areas = np.array(list(map(cv2.contourArea,contours)))

    if len(areas) == 0 or np.max(areas) / (h*w) < AREA_RATIO_THRESHOLD:
        # 見つからなかったらNoneを返す
        return None
    else:
        # 面積が最大の塊の重心を計算し返す
        max_idx = np.argmax(areas)
        max_area = areas[max_idx]
        result = cv2.moments(contours[max_idx])
        p = int(result["m10"]/result["m00"])
        q = int(result["m01"]/result["m00"])
        return (p,q)

def pos_to_frequency1(x): #x座標に応じて数字を返す
    #pos_listに、境界となる座標を格納
    pos_list = [12,36,60,84,108,132,156,180,204,228,252,276,300,324,348,372,396,420,444,468,492,516,540,564,588,612]
    for i in range(13):
        if x>=pos_list[2*i] and x<=pos_list[2*i+1]:
            ans = i
            break
        else:
            ans = 50
    return ans

def pos_to_frequency2(y): #y座標に応じて数値を返す yに関しての条件はゆるい
    if y>=320:
        return(100)
    elif y<=150:
        return(0)
    else:
        return(100000)

def circle_on_the_key(a,b): #a=i,b=num
    #弾いた音に対応する鍵盤に円が描画されるようにする関数
    x_list =  [55,72,89,106,122,156,170,189,207,220,241,255,285,304,317,336,350,383,400,417,435,450,469,484,517]  #それぞれの鍵盤の中心のx座標              
    if (b>100 and (b%100 == 1 or b%100 == 3 or b%100 == 6 or b%100 == 8 or b%100 == 10)) or b==0 or b==2 or b==5 or b==7 or b==9:
        y = 250
    else:
        y = 300
    x = x_list[a]
    cv2.circle(frame,(x,y),10,(255,255,255),-1)

def sound_by_blue_top(a_list,b_list): #青色が検出された座標に応じて音を奏でる、a_list =file_list、b_list = doremi_list
    cv2.circle(frame,pos,10,(255,255,255),-1)

    num = pos_to_frequency1(pos[0])+pos_to_frequency2(pos[1])

    for i in range(len(num_list)):

        if num == num_list[i]:

            if (i<=12 and normal_doremi[i] == 0) or (i>12 and high_doremi[i-12] == 0):
                target_sound = AudioSegment.from_file(a_list[i],"wav")
                play(target_sound)

                global Sound
                Sound = b_list[i]

                global sum
                if before_playing == False:
                    sum+=1
                
                if i<=12:
                    for j in range(13):
                        normal_doremi[j] = 0
                    normal_doremi[i] = 1
                else:
                    for j in range(13):
                        high_doremi[j] = 0
                    high_doremi[i-12] = 1
            
            circle_on_the_key(i,num)

def sound_by_key(a_list,b_list): #キーボードを鍵盤に見立てて音を奏でる

    for i in range(13):
        normal_doremi[i] = 0
        high_doremi[i] = 0
    
    key_list = ['z','s','x','d','c','v','g','b','h','n','j','m','w','3','e','4','r','t','6','y','7','u','8','i','o'] #音に対応するキーのリスト
    key = cv2.waitKey(5) & 0xFF
    for i in range(len(key_list)):
        if key == ord(key_list[i]):
            target_sound = AudioSegment.from_file(a_list[i],"wav") #どのキーを押したかによって音が決まる
            play(target_sound)
            global Sound
            Sound = b_list[i] #最後に弾いた音として出した音を記録する
            global sum
            if before_playing == False:
                sum+=1 #ゲームが始まっている場合は、弾いた音の数として加算する
            num = num_list[i] #ciecle_on_the_keyにnumが因数として必要なので、対応させる
            circle_on_the_key(i,num) #弾いた音に対応するキーに画面上で円を描写
def play_sound(): 

    if pos is not None:  #青色があるときは、それによって音を奏でる
        sound_by_blue_top(file_list,doremi_list)
    
    if pos is None:  #青が見つからないときは、キーボードからの入力にしたがう。
        sound_by_key(file_list,doremi_list)

def game_start(): #Aキーが押されたら、曲選択の説明画面を表示する
    cv2.rectangle(frame, (0,90), (635, 130), (186,172,0),thickness = -1)
    cv2.putText(frame,'Choose your favorite song',(90,120),cv2.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),thickness = 2)
    cv2.putText(frame,'1. Twinkle Twinkle Little Star',(190,350),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)
    cv2.putText(frame,'2. The Bear Song',(235,370),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)
    cv2.putText(frame,"3. Love's Greeting",(227,390),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)
    cv2.putText(frame,'4. The Only Flower in the World',(175,410),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)
    cv2.putText(frame,'5. kimi wa rokku o kikanai',(212,430),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)
    cv2.putText(frame,"6. Pretender",(260,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),thickness = 2)

def song_select():  #これを繰り返しにするのは事情により難しそう どの曲を選択したかを返す関数　曲を選択した後、ゲーム開始前か否かを示す変数を書き換える
    key2 = cv2.waitKey(20) & 0xFF
    if key2 == ord('1'):
        p = 1
        global before_playing
        before_playing = False
    elif key2 == ord('2'):
        p = 2
        global before_playing
        before_playing = False
    elif key2 == ord('3'):
        p = 3
        global before_playing
        before_playing = False
    elif key2 == ord('4'):
        p = 4
        global before_playing
        before_playing = False
    elif key2 == ord('5'):
        p = 5
        global before_playing
        before_playing = False
    elif key2 == ord('6'):
        p = 6
        global before_playing
        before_playing = False
    elif key2 == ord('7'):
        p = 7
        global before_playing
        before_playing = False
    else:
        p = 8
    return p

def choice_list(x,list1,list2,list3,list4,list5,list6,list7): #x=check  何番を選んだかに応じて、どのリストを正解として照らし合わせるべきかを示す
    answer = [list1,list2,list3,list4,list5,list6,list7]
    return answer[x-1]

def compare(check): #弾かれた音と、正解リストの音を比較していく
    if Sound != "null":
        global fit
        global unfit
        checklist = choice_list(check,Answer_list,Answer_list2,Answer_list3,Answer_list4,Answer_list5,Answer_list6,Answer_list7)
        print(checklist)
        if Sound == checklist[fit]:
            fit += 1
            if fit+unfit > sum:
                fit = sum - unfit
        else:
            unfit += 1
            if fit + unfit > sum:
                 unfit = sum - fit
        print(fit)
        print(unfit)
        print(before_playing)
        print(sum)

def check_result():  #ミスの回数に対応した画像を表示する関数
    global sum
    global fit
    if fit==sum:
        img = cv2.imread('IMG_7556.jpg',-1)
    elif sum-fit<=5:
        img = cv2.imread('IMG_7552.jpg',-1)
    elif sum-fit<=10:
        img = cv2.imread('IMG_7554.jpg',-1)
    else:
        img = cv2.imread('IMG_7555.jpg',-1)
    resized_img = cv2.resize(img,(width,height))

    cv2.imshow('result',resized_img)
    sum = 0
    fit = 0
    unfit = 0

if __name__ == '__main__':  
      
    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    w = width//4
    h = height//4

    hoge = False
    p = 8
    while True:

        ret,frame = cap.read() # カメラ映像の読み込み

        frame = cv2.flip(frame,1) #　左右反転(弾くときの手の動かす方向と、画面上の移動方向を一致させる)

        message() #ゲーム前かどうかに応じて適切なメッセージを表示する

        pos = find_specific_color(
            frame,
            AREA_RATIO_THRESHOLD,
            LOW_COLOR_BLUE,
            HIGH_COLOR_BLUE
        )

        play_sound()

        key = cv2.waitKey(5) & 0xFF
        if key == ord('a'):
            game_start()
            hoge = True
        elif key == ord('q'):
            break

        if hoge:
            game_start()
            check = song_select()
            if check<=7:
                hoge = False

        for i in range(1,8):
            if check == i and fit != len(choice_list(i,Answer_list,Answer_list2,Answer_list3,Answer_list4,Answer_list5,Answer_list6,Answer_list7)):
                compare(check)
            elif check == i and fit == len(choice_list(i,Answer_list,Answer_list2,Answer_list3,Answer_list4,Answer_list5,Answer_list6,Answer_list7)):
                check_result()

        #カメラからとりこんだ画像と、ピアノの鍵盤の画像、星空の画像を透過させて合成して表示
        img1 = cv2.imread('star_sky.png')
        resized_img1 = cv2.resize(img1,(640,480))
        img2 = cv2.imread('IMG_7569.jpg')
        resized_img2 = cv2.resize(img2,(640,480))
        alpha = 0.4
        blended_base = cv2.addWeighted(resized_img1, alpha, resized_img2, 1 - alpha, 0)  # img1 * 0.4 + img2 * 0.6
        blended_final = cv2.addWeighted(frame, 0.3, blended_base, 0.7, 0) 
        blended_final = cv2.resize(blended_final, dsize=(1280, 960))
        cv2.imshow('piano',blended_final)
        
    cap.release()
    cv2.destroyAllWindows()