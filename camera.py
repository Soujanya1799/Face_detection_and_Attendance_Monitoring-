import cv2
import threading
import time
import numpy as np
import json

lock = threading.Lock()

class CameraThread(threading.Thread):
    def __init__(self, camera_serial_number, rtsp_link, frame_dictionary,fps):
        assert fps > 0, "FPS is not greater than 0"
        super(CameraThread, self).__init__()
        self.cam_serial_num = camera_serial_number
        self.rtsp_url = rtsp_link
        self.frame_dict = frame_dictionary
        self.running = True
        self.cap = None
        self.daemon = True
        self.sleep_duration = 1/fps #to match the camera fps

    def run(self):
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"Error: Failed to connect to camera {self.cam_serial_num}")
            return

        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with lock: #to make sure frame_dict is only accessed by respective thread
                    self.frame_dict[self.cam_serial_num] = frame
            else:
                print(f"Error: Failed to receive frame from camera {self.cam_serial_num}")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)
            time.sleep(self.sleep_duration)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False

def load_camera_config(json_file):
    try:
        with open(json_file, 'r', encoding="utf-8") as f:
            config = json.load(f)
        return config
    except FileNotFoundError as e:
        raise FileNotFoundError("Config file 'cameras.json' not found.") from e