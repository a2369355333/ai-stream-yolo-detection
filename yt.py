import cv2
from time import time
from pytube import YouTube
import yt_dlp
from ultralytics import YOLO
import pandas as pd

def process_youtube_stream(youtube_url):
    # 載入YOLO模型 (改成ultralytics最新API)
    model = YOLO('yolov5s.pt')
    
    # 只偵測人 (class 0)
    model.classes = [0]
    
    # 透過yt-dlp取得串流網址
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            stream_url = info['url']
    except Exception as e:
        print(f"yt-dlp error: {e}")
        try:
            yt = YouTube(youtube_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            stream_url = stream.url
        except Exception as e2:
            print(f"pytube error: {e2}")
            print("Failed to get stream URL. Please check your internet connection and YouTube URL.")
            return
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error opening video stream")
        return
    
    frame_count = 0
    start_time = time()
    fps = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        elapsed = time() - start_time
        if elapsed >= 1:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time()
        
        # OpenCV讀的影像是BGR，要轉成RGB給YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = model(frame_rgb)
        result = results[0]
        
        # 將結果轉DataFrame 
        df = result.boxes.data.cpu().numpy() if hasattr(result, 'boxes') else None
        df = result.to_df() if hasattr(result, 'to_df') else None

        if df is None or df.empty:
            persons = pd.DataFrame() 
        else:
            persons = df[df['class'] == 0]
        
        # 繪製框和標籤
        for _, person in persons.iterrows():
            box = person['box']
            xmin, ymin = int(box['x1']), int(box['y1'])
            xmax, ymax = int(box['x2']), int(box['y2'])
            confidence = person['confidence']
            
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            label = f"Person: {confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"People: {len(persons)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('YouTube Livestream Person Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=NxcL7gydx1U"
    process_youtube_stream(youtube_url)
