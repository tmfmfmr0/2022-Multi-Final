import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import joblib
from moviepy.editor import *
import math
import sys
import time
import imutils
import os

model = joblib.load('./models/RandomForestFinal.pkl')

# 포즈 감지 모델 초기화
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7,
                          min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ========================= 포즈감지 및 각도측정 =========================

def detectPose(image_pose, pose, draw=False, display=False):
    
    original_image = image_pose.copy()
    
    image_in_RGB = cv2.cvtColor(image_pose, cv2.COLOR_BGR2RGB)
    
    resultant = pose.process(image_in_RGB)

    if resultant.pose_landmarks and draw:    

        mp_drawing.draw_landmarks(image=original_image, landmark_list=resultant.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,255),
                                                                               thickness=3, circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49,125,237),
                                                                               thickness=2, circle_radius=2))

    if display:
            
            plt.figure(figsize=[22,22])
            plt.subplot(121);plt.imshow(image_pose[:,:,::-1]);plt.title("Input Image");plt.axis('off');
            plt.subplot(122);plt.imshow(original_image[:,:,::-1]);plt.title("Pose detected Image");plt.axis('off');

    else:
        
        return original_image, resultant

def norm(data):
    data = np.array(data)
    x = data.T[0]
    y = data.T[1]
    z = data.T[2]
    x_norm = (x - min(x)) / (max(x) - min(x))
    y_norm = (y - min(y)) / (max(y) - min(y))
    z_norm = (z - min(z)) / (max(z) - min(z))
    
    return (x_norm.tolist(), y_norm.tolist(), z_norm.tolist())

def link_vector(land):
    link_keypoint = [(0, 1),
        (1, 3),
        (3,	5),
        (5,	7),
        (5,	9),
        (5,	11),
        (1, 13),
        (13, 15),
        (15, 17),
        (17, 19),
        (17, 21),
        (0, 2),
        (2, 4),
        (4, 6),
        (4, 8),
        (4, 10),
        (4, 12),
        (2, 14),
        (14, 16),
        (16, 18),
        (18, 20),
        (18, 22)]
    
    a = []
    for link in link_keypoint:
        x = land[0][link[0]] - land[0][link[1]]
        y = land[1][link[0]] - land[1][link[1]]
        z = land[2][link[0]] - land[2][link[1]]
        a.append((x, y, z))
    return a

def angle_vector(land):
    
    angle_keypoint=[
        (0, 1, 3),
        (1, 3, 5),
        (3, 5, 9),
        (1, 13, 15),
        (13, 15, 17),
        (15, 17, 19),
        (15, 17, 21),
        (0, 2, 4),
        (2, 4, 6),
        (4, 6, 10),
        (2, 14, 16),
        (14, 16, 18),
        (16, 18, 20),
        (16, 18, 22)]
    
    a = []
    for angle in angle_keypoint:
        x = np.array([land[0][angle[0]] - land[0][angle[1]], land[1][angle[0]] - land[1][angle[1]], land[2][angle[0]] - land[2][angle[1]]])
        y = np.array([land[0][angle[1]] - land[0][angle[2]], land[1][angle[1]] - land[1][angle[2]], land[2][angle[1]] - land[2][angle[2]]])
        
        분자 = np.dot(x, y)
        분모 = np.sqrt(x.dot(x)) * np.sqrt(x.dot(x))
        try:
            a.append(math.acos(분자 / 분모))
        except:
            a.append(0)
    return (a)

def pose_feature(link, angle):
    산술평균_링크 = [np.mean(link.T[0]), np.mean(link.T[1]), np.mean(link.T[2])]
    표준편차_링크 = [np.std(link.T[0]), np.std(link.T[1]), np.std(link.T[2])]
    제곱평균_링크 = [np.mean(link.T[0]**2), np.mean(link.T[1]**2), np.mean(link.T[2]**2)]
    
    산술평균_앵글 = np.mean(angle)
    표준편차_앵글 = np.std(angle)
    제곱평균_앵글 = np.mean(angle)
    
    return(산술평균_링크+표준편차_링크+제곱평균_링크+[산술평균_앵글]+[표준편차_앵글]+[제곱평균_앵글])


#========================= 일치율 측정 =========================


def get_sim(target, user):
    target_landmarks = target.pose_world_landmarks.landmark
    target_lm = [(i.x, i.y, i.z) for num, i in enumerate(target_landmarks) if num not in range(1, 11)]
    target_norm = norm(target_lm)
    target_link_vector = link_vector(target_norm)
    target_angle_vector = angle_vector(target_norm)
    
    user_landmarks = user.pose_world_landmarks.landmark
    user_lm = [(i.x, i.y, i.z) for num, i in enumerate(user_landmarks) if num not in range(1, 11)]
    user_norm = norm(user_lm)
    user_link_vector = link_vector(user_norm)
    user_angle_vector = angle_vector(user_norm)
    
    link_diff = np.array(target_link_vector) - user_link_vector
    angle_diff = np.array(target_angle_vector) - user_angle_vector
    feature = pose_feature(link_diff, angle_diff); feature = np.array(feature).reshape(1, -1)
    similarity = model.predict_proba(feature)[0][1]
    
    return similarity


#===================================================

def make_result(video1, video2):
    
    video_clip = VideoFileClip(video1)
    video_clip2 = VideoFileClip(video2)
    video2len = video_clip2.duration
    video_clip3 = video_clip2.subclip(3, float(video2len))
    
    video_len = min(video_clip.duration, video_clip3.duration)
    
    audioclip = video_clip.audio
    audioclip.write_audiofile('./static/sim_out/target.mp3')
    
    w = int(video_clip2.w)
    h = int(video_clip2.h)
    fourcc = cv2.VideoWriter_fourcc(*"x264")
    out = cv2.VideoWriter('./static/sim_out/output.mp4', fourcc, 1/0.03, (w,h))
    
    ss = []
    sc = []
    text = ''
    
    for num, i in enumerate(np.arange(0, video_len, 0.03)):   # (0, videl_len, 0.03)이 원본
        
        img = video_clip.get_frame(i)
        img2 = video_clip2.get_frame(i)
        
        img_user = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        if (num+1) % 20 == 0:
            ss_mean = round(np.mean([i*100 for i in ss[num-18:] if i is not None]))
            sc.append(ss_mean)
            score = ('BAD' if ss_mean <= 30 else 'MISS' if ss_mean <= 40 else 'GOOD' if ss_mean <= 50 else 'VERY GOOD' if ss_mean <= 60 else 'PERPECT')
            text = f'{score}'

        try:
            result = detectPose(img, pose_video)[1]
            result2 = detectPose(img2, pose_video)[1]
            
            similarity = get_sim(result, result2)
            ss.append(similarity)

            cv2.putText(img_user,  text, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (199, 114, 255), 2, cv2.LINE_AA)        # BGR
            cv2.imshow("target", img_user)
            
        except:
            ss.append(None)
            
            cv2.putText(img_user,  text, (0, 50), cv2.FONT_HERSHEY_DUPLEX, 2, (199, 114, 255), 2, cv2.LINE_AA)        # BGR
            cv2.imshow("target", img_user)


        out.write(img_user) #프레임 쓰기
    
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    total_score = round(np.mean(sc))
    text_score = ('F' if total_score <= 30 else 'D' if total_score <= 40 else 'C' if total_score <= 50 else 'B' if total_score <= 60 else 'A')
    score_img = cv2.imread(f'./static/score/{text_score}.PNG',1)
    score_img = imutils.resize(score_img, width=w)

    st = round((score_img.shape[0] - h)/2)
    score_img = score_img[st:st+h]

    for i in range(0, 100):
        out.write(score_img) #프레임 쓰기

    cv2.destroyAllWindows()
    out.release()
    
    return ss
