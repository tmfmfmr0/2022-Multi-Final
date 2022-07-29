from flask import Flask, redirect, render_template, request, send_file
from flask import current_app
from utils import dance
import cv2
from werkzeug.utils import secure_filename
import os, math, pandas as pd, numpy as np, matplotlib.pyplot as plt

app = Flask(__name__)

path_dance = './static/dance'
path_dance2 = './static/dance2'
path_user_dance = './static/user_dance'
path_user_360 = './static/user_360'
path_background = './static/background'


@app.route('/')
def index():
    menu = {'home': 1, 'menu1': 0, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 0}

    return render_template('index.html', menu=menu)


# 춤 일치도 분석
@app.route('/menu1', methods=['GET', 'POST'])
def menu1():
    menu = {'home': 0, 'menu1': 1, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 0}
    
    if request.method == 'GET' :
        # 저장되어 있는 파일 목록 가져오기
        list_dance = os.listdir(path_dance)
        list_user_dance = os.listdir(path_user_dance)
        # 파일 목록을 리스트에 딕셔너리 형식으로 넣기
        dance_options, user_dance_options = [], []
        for i in range(len(list_dance)):
            dance_options.append({'disp': list_dance[i][:-4] , 'val':list_dance[i] })
        for i in range(len(list_user_dance)):
            user_dance_options.append({'disp': list_user_dance[i][:-4] , 'val':list_user_dance[i] }) 
        dance_options.append({'disp': '직접추가', 'val':'direct1'})
        user_dance_options.append({'disp': '직접추가', 'val':'direct2'})

        return render_template('menu1.html', menu=menu, dance_options=dance_options, 
                                user_dance_options=user_dance_options)
    
    else :    # request.method == 'POST'
        dance_options = request.form['dance_option']
        if not os.path.exists(os.path.join(current_app.root_path, 'static/dance')):
            os.makedirs(os.path.join(current_app.root_path, 'static/dance'))
        if dance_options == 'direct1':
            dance_mp4 = request.files['dance_upload']
            dance_file = 'static/dance/' + dance_mp4.filename
            dance_mp4.save(dance_file)
        else:
            dance_file = 'static/dance/' + dance_options

        user_dance_options = request.form['user_dance_option']
        if not os.path.exists(os.path.join(current_app.root_path, 'static/user_dance')):
            os.makedirs(os.path.join(current_app.root_path, 'static/user_dance'))
        if user_dance_options == 'direct2':
            user_dance_mp4 = request.files['user_dance_upload']
            user_dance_file = 'static/user_dance/' + user_dance_mp4.filename
            user_dance_mp4.save(user_dance_file)
        else:
            user_dance_file = 'static/user_dance/' + user_dance_options
        
        return render_template('sim_spinner.html', menu=menu, dance_file=dance_file, user_dance_file=user_dance_file)

# 일치도 녹화화면
@app.route('/menu1_rec', methods=['GET', 'POST'])
def menu1_rec():
    menu = {'home': 0, 'menu1': 0, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 0}
    
    if request.method == 'GET' :
        # 저장되어 있는 파일 목록 가져오기
        list_dance = os.listdir(path_dance)
        # 파일 목록을 리스트에 딕셔너리 형식으로 넣기
        dance_options  = []
        for i in range(len(list_dance)):
            dance_options.append({'disp': list_dance[i][:-4] , 'val':list_dance[i] })
        dance_options.append({'disp': '직접추가', 'val':'direct1'})

        return render_template('menu1_rec.html', menu=menu, dance_options=dance_options)

    else :    # request.method == 'POST'
        
        user_dance_mp4 = request.files['video_blob']
        user_dance_file = 'static/user_dance_record/raw_video.avi'
        user_dance_mp4.save(user_dance_file)
        
        return '0'

# 일치도 결과화면
@app.route('/menu1_res', methods=['GET', 'POST'])
def menu1_res():
    menu = {'home': 0, 'menu1': 0, 'menu2': 0, 'menu1_res': 1, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 0}
    
    
    return render_template('menu1_res.html', menu=menu)

    
@app.route('/video_rec_proc', methods=['POST'])
def video_rec_proc():
    menu = {'home': 0, 'menu1': 0, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 1, 'menu2_res': 0}
    
    raw_file = 'static/user_dance_record/raw_video.avi'
    cap = cv2.VideoCapture(raw_file)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # 또는 cap.get(3)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
    fps = cap.get(cv2.CAP_PROP_FPS)             # 또는 cap.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'x264')    # 코덱 정의, *'mp4v' == 'm', 'p', '4', 'v', 또는 *'DIVX'
    out = cv2.VideoWriter('static/user_dance/user_rec_video.mp4', fourcc, fps, (int(width), int(height))) # VideoWriter 객체
    while True:
        ret, img = cap.read()
        if ret:
            out.write(cv2.flip(img, 1))     # 좌우 반전 시켜서 원 위치로 환원
            cv2.waitKey(1)                 # 30 fps
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    dance_options = request.form['dance_option']
    if not os.path.exists(os.path.join(current_app.root_path, 'static/dance')):
        os.makedirs(os.path.join(current_app.root_path, 'static/dance'))
    if dance_options == 'direct1':
        dance_mp4 = request.files['dance_upload']
        dance_file = 'static/dance/' + dance_mp4.filename
        dance_mp4.save(dance_file)
    else:
        dance_file = 'static/dance/' + dance_options
        
    user_dance_file = 'static/user_dance/user_rec_video.mp4'

    return render_template('sim_spinner.html', menu=menu, dance_file=dance_file, user_dance_file=user_dance_file)

# 일치도 처리과정
@app.route('/menu1_proc', methods=['POST'])
def menu1_proc():
    menu = {'home': 0, 'menu1': 0, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 1, 'video_rec_proc': 0, 'menu2_res': 0}
    
    dance_file = request.form['dance_file']
    user_dance_file = request.form['user_dance_file']
    
    sim = dance.make_result(dance_file, user_dance_file)
    
    return render_template('menu1_res.html', menu=menu, dance_file=dance_file, user_dance_file=user_dance_file, sim=sim)
    
        
# 춤 입히기
@app.route('/menu2', methods=['GET', 'POST'])
def menu2():
    menu = {'home': 0, 'menu1': 0, 'menu2': 1, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 0}
    
    if request.method == 'GET':
        # 저장되어 있는 파일 목록 가져오기
        list_dance = os.listdir(path_dance2)
        list_user_360 = os.listdir(path_user_360)
        list_background = os.listdir(path_background)
        # 파일 목록을 리스트에 딕셔너리 형식으로 넣기
        dance_options, user_360_options, background_options = [], [], []
        for i in range(len(list_dance)):
            dance_options.append({'disp': list_dance[i][:-4] , 'val':list_dance[i] })
        for i in range(len(list_user_360)):
            user_360_options.append({'disp': list_user_360[i][:-4] , 'val':list_user_360[i] })
        for i in range(len(list_background)):
            background_options.append({'disp': list_background[i][:-4] , 'val':list_background[i] })
        dance_options.append({'disp': '직접추가', 'val':'direct1'})
        user_360_options.append({'disp': '직접추가', 'val':'direct2'})
        background_options.append({'disp': '직접추가', 'val':'direct3'})

        return render_template('menu2.html', menu=menu, dance_options=dance_options,
                                user_360_options=user_360_options, 
                                background_options=background_options)
        
    else:    # request.method == 'POST'
        dance_options = request.form['dance_option']
        if not os.path.exists(os.path.join(current_app.root_path, 'static/dance2')):
            os.makedirs(os.path.join(current_app.root_path, 'static/dance2'))
        if dance_options == 'direct1':
            dance_mp4 = request.files['dance_upload']
            dance_file = 'static/dance2/' + dance_mp4.filename
            dance_mp4.save(dance_file)
        else:
            dance_file = 'static/dance2/' + dance_options

        user_360_options = request.form['user_360_option']
        if not os.path.exists(os.path.join(current_app.root_path, 'static/user_360')):
            os.makedirs(os.path.join(current_app.root_path, 'static/user_360'))
        if user_360_options == 'direct2':
            user_360_mp4 = request.files['user_360_upload']
            user_360_file = 'static/user_360/' + user_360_mp4.filename
            user_360_mp4.save(user_360_file)
        else:
            user_360_file = 'static/user_360/' + user_360_options
            
        background_options = request.form['background_option']
        if not os.path.exists(os.path.join(current_app.root_path, 'static/background')):
            os.makedirs(os.path.join(current_app.root_path, 'static/background'))
        if background_options == 'direct3':
            bg_png = request.files['background_upload']
            bg_file = 'static/background/' + bg_png.filename
            bg_png.save(bg_file)
        else:
            bg_file = 'static/background/' + background_options
            
        return render_template('/menu2_res.html', menu=menu, dance_file=dance_file, user_360_file=user_360_file, bg_file=bg_file)

@app.route('/menu2_res', methods=['POST'])
def menu2_res():
    menu = {'home': 0, 'menu1': 0, 'menu2': 0, 'menu1_res': 0, 'menu1_proc': 0, 'video_rec_proc': 0, 'menu2_res': 1}
    
    return render_template('menu2_res.html')






if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5500, debug=True)
