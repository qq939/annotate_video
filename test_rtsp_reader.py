import unittest
import threading
import time
from rtsp_reader import read_rtsp_stream

class TestRTSPReader(unittest.TestCase):
    def test_read_rtsp_stream_timeout(self):
        """
        测试能否正常拉取流，或者在指定时间内（例如 5 秒）退出而不被永久阻塞
        """
        # 我们使用 threading 和 event 来控制超时
        result = []
        def target():
            try:
                # read_rtsp_stream 返回是否成功获取到至少一帧
                res = read_rtsp_stream(max_frames=1)
                result.append(res)
            except Exception as e:
                result.append(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()

        # 超时机制：7秒后如果没有返回，则认为测试失败（被阻塞），给 OpenCV 5 秒超时留出余量
        thread.join(timeout=7.0)
        
        if thread.is_alive():
            self.fail("测试超时，拉流函数可能发生了阻塞")
            
        self.assertTrue(len(result) > 0, "函数未返回任何结果")
        # 即使连不上，由于是 RTSP 地址可能不可达，也应该不阻塞地返回 False 或抛出异常
        # 这里只要函数不阻塞即算测试通过

if __name__ == '__main__':
    unittest.main()
