import cv2
import os

# ==========================================
# GLOBAL PARAMETERS (全局参数)
# ==========================================
# RTSP_URL 在本文件的第 21 行作为参数传给 cv2.VideoCapture() 用于拉流
RTSP_URL = "rtsp://192.168.8.102:8554/ams/live"

# 设置环境变量确保 OpenCV 拉流时有超时机制（单位：微秒，5000000=5秒）
# 在本文件的第 19 行前设置，避免因 RTSP 不可达导致的永久阻塞
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000"

def read_rtsp_stream(max_frames=None):
    """
    拉取 RTSP 流并读取。
    为了简洁，只使用 cv2.VideoCapture。
    max_frames 用于测试目的，限制读取的帧数。
    """
    # 21行: 使用 RTSP_URL 建立连接
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print(f"无法打开 RTSP 流: {RTSP_URL}")
        return False
    
    frames_read = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收帧 (流可能已结束或断开连接)")
            break
            
        # 显示帧，如果是服务器环境无界面可能会报错，此处仅做示例
        # cv2.imshow('RTSP Stream', frame)
        frames_read += 1
        
        # 针对测试环境，如果指定了最大帧数则退出
        if max_frames and frames_read >= max_frames:
            break
            
        # 按 'q' 键退出
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
    cap.release()
    # cv2.destroyAllWindows()
    return frames_read > 0

if __name__ == "__main__":
    read_rtsp_stream()
