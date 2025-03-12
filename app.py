from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import threading

app = Flask(__name__)
CORS(app)  # 启用CORS支持所有域

# 初始化 MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 创建线程池
executor = ThreadPoolExecutor(max_workers=4)

# 视频处理配置
VIDEO_CONFIG = {
    'max_frames': 90,
    'frame_interval': 3,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# 创建结果缓存
result_cache = {}
cache_lock = threading.Lock()

def download_video(url):
    """下载视频到临时文件"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        temp_dir = '/tmp' if os.path.exists('/tmp') else tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'video_{time.time()}.mp4')
        
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return temp_path
    except Exception as e:
        print(f"下载视频失败: {str(e)}")
        return None

def extract_frames(video_path):
    """提取视频帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算采样间隔
    sample_interval = max(1, total_frames // VIDEO_CONFIG['max_frames'])
    
    frame_count = 0
    while cap.isOpened() and len(frames) < VIDEO_CONFIG['max_frames']:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_interval == 0:
            # 调整帧大小以加快处理速度
            frame = cv2.resize(frame, (640, 480))
            frames.append(frame)
        frame_count += 1
        
    cap.release()
    return frames

def process_frames(frames):
    """处理视频帧并进行手语识别"""
    try:
        with mp_holistic.Holistic(
            min_detection_confidence=VIDEO_CONFIG['min_detection_confidence'],
            min_tracking_confidence=VIDEO_CONFIG['min_tracking_confidence'],
            model_complexity=0  # 使用较轻量级的模型
        ) as holistic:
            landmarks_sequence = []
            
            for frame in frames:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                
                if results.pose_landmarks:
                    pose = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
                    landmarks_sequence.append(pose)
                    
            return analyze_landmarks(landmarks_sequence)
    except Exception as e:
        print(f"处理帧失败: {str(e)}")
        return None

def analyze_landmarks(landmarks_sequence):
    """分析关键点序列并返回识别结果"""
    if not landmarks_sequence:
        return "未检测到有效的手语动作"
    
    # TODO: 实现实际的手语识别算法
    return "手语识别结果：你好"

def process_video_task(video_path):
    """异步处理视频任务"""
    try:
        frames = extract_frames(video_path)
        if not frames:
            return None
        return process_frames(frames)
    except Exception as e:
        print(f"视频处理任务失败: {str(e)}")
        return None
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        video_url = data.get('video_url')
        
        if not video_url:
            return jsonify({
                "code": 400,
                "error": "缺少视频URL"
            })

        # 检查缓存
        with cache_lock:
            if video_url in result_cache:
                return jsonify({
                    "code": 200,
                    "data": {
                        "result": result_cache[video_url],
                        "processed": int(time.time()*1000),
                        "cached": True
                    }
                })

        # 下载视频
        video_path = download_video(video_url)
        if not video_path:
            return jsonify({
                "code": 500,
                "error": "视频下载失败"
            })

        try:
            # 使用线程池处理视频
            future = executor.submit(process_video_task, video_path)
            result = future.result(timeout=25)

            if not result:
                raise Exception("视频处理失败")

            # 缓存结果
            with cache_lock:
                result_cache[video_url] = result

            return jsonify({
                "code": 200,
                "data": {
                    "result": result,
                    "processed": int(time.time()*1000)
                }
            })
        finally:
            # 清理临时文件
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
            except:
                pass

    except Exception as e:
        print(f"识别错误: {str(e)}")
        return jsonify({
            "code": 500,
            "error": str(e)
        })

# 用于Vercel等平台的处理函数
def handler(request):
    with app.request_context(request):
        return app.full_dispatch_request()

# 本地开发服务器
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
	
	