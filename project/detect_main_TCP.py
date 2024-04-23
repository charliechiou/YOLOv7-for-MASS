import sys
import socket
import detect_gui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import threading
import torch
import numpy as np
import os
import librosa
import librosa.display
import sounddevice as sd
import pyaudio
import noisereduce as nr
import matplotlib.pyplot as plt
import warnings
sys.path.append(r"C:\\chiou\\yolov7")
import time
import detect_with_API

class MainWindows(QFrame, detect_gui.Ui_main):
    x_position_signal = pyqtSignal(str)
    y_position_signal = pyqtSignal(str)
    degree_signal = pyqtSignal(str)
    direction_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.start_time = time.time()
        print(self.start_time)

        self.video = threading.Thread(target=self.opencv)
        self.video.start()
        
        self.x_position_signal.connect(self.x_display)
        self.y_position_signal.connect(self.y_display)
        self.degree_signal.connect(self.degree_show_display)
        self.direction_signal.connect(self.direction_show)

        self.record_button.clicked.connect(self.audio)

        self.tcp = threading.Thread(target = self.tcp)
        self.tcp.start()

        self.audio_start = False
        self.soundDirection = 'none'

    #tcp
    def tcp(self):
        HOST = '127.0.0.1'
        PORT = 8088

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)

        print('server start at: %s:%s' % (HOST, PORT))
        print('wait for connection...')
        print('------------------------')

        while True:
            conn, addr = s.accept()
            print('connected by ' + str(addr))
            print('-----------------------------')
            while True:
                if self.audio_start == True:
                    conn.send(self.soundDirection.encode())
                    self.audio_start = False

                if self.direction == 'Right' and self.audio_start == False:
                    conn.send(self.degree.encode())

                if self.direction == 'Left' and self.audio_start == False:
                    conn.send(("-"+self.degree).encode())                


                time.sleep(0.5)


    ## number detect ##
    def x_display(self, x):
        self.x_position_display.setText(x)

    def y_display(self, y):
        self.y_position_display.setText(y)

    def degree_show_display(self, degree):
        self.degree_display.setText(degree)

    def direction_show(self, direction):
        self.direction_display.setText(direction)

    def closeOpenCV(self, a):
        global ocv
        ocv = False

    def opencv(self):
        cam_feed = cv2.VideoCapture(1)
        # cam_feed = cv2.VideoCapture(
        #     "C:\chiou\\number_video\\717152631.778620.mp4")

        if not (cam_feed.isOpened()):
            print('Could not open camera')
        else:
            print('Camera opened')

        cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        a = detect_with_API.detectapi(
            weights="C:\chiou\yolov7\\runs\\train\medium_data\weights\\best.pt", img_size=640)
        self.x = "none"
        self.y = "none"
        self.degree = "none"
        self.direction = "none"
        while ocv:
            _, img = cam_feed.read()
            height, width, channel = img.shape
            bytePerline = channel * width
            with torch.no_grad():
                result, names = a.detect([img])
                img = result[0][0]
                for cls, (x1, y1, x2, y2), conf in result[0][1]:
                    # print(names[cls],x1,y1,x2,y2,conf)
                    if names[cls] == self.num_chose_control.currentText():
                        self.x = str((x1+x2)/2/640)
                        self.y = str((y1+y2)/2/480)
                        if (x1+x2)/2 > 320:
                            self.degree = str(round(203.53*(((x1+x2)/2/640)-0.5)*(((x1+x2)/2/640)-0.5)+32.579*(((x1+x2)/2/640)-0.5)+0.0794,3))
                            self.direction = "Right"
                        if (x1+x2)/2 < 320:
                            self.degree = str(round(203.53*(0.5-((x1+x2)/2/640))*(0.5-((x1+x2)/2/640))+32.579*(0.5-((x1+x2)/2/640))+0.0794,3))
                            self.direction = "Left"
                    else:
                        self.x = "none"
                        self.y = "none"
                        self.degree = "none"
                        self.direction = "none"

                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                    cv2.putText(
                        img, names[cls], (x1, y1-30), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), thickness=2)
            qimg = QImage(img, width, height, bytePerline,
                          QImage.Format_BGR888)

            self.cam_video.setPixmap(QPixmap.fromImage(qimg))
            self.x_position_signal.emit(self.x)
            self.y_position_signal.emit(self.y)
            self.degree_signal.emit(self.degree)
            self.direction_signal.emit(self.direction)

    ## audio detect ##
    def audio(self):
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # Recording parameters
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = self.time_control.value()

        # Initialize recording object
        p = pyaudio.PyAudio()

        # Open audio stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        print("================================================")
        print(
            f"record second:{RECORD_SECONDS} \n threshold:{self.threshold_control.value()}")
        print("Recording...")

        frames = []

        # Record audio data
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("Recording finished.")

        # Close audio stream
        stream.stop_stream()
        stream.close()

        # Close recording object
        p.terminate()

        # Convert recorded data to a numpy array
        mic_audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        # Convert integers to floats and normalize
        mic_audio = mic_audio.astype(np.float32) / 32767.0

        # Apply pre-emphasis to the microphone audio
        pre_emphasized_audio = np.append(
            mic_audio[0], mic_audio[1:] - 0.97 * mic_audio[:-1])

        # Apply noise reduction using spectral subtraction
        reduced_audio = nr.reduce_noise(y=pre_emphasized_audio, sr=RATE)

        # Calculate MFCCs of the noise-reduced audio
        mic_mfccs_reduced = librosa.feature.mfcc(
            y=reduced_audio, sr=RATE, n_mfcc=13)

        # Extract the frequency range from 100 to 5000 Hz
        min_freq = 100
        max_freq = 4500
        min_idx = int(min_freq * CHUNK / RATE)
        max_idx = int(max_freq * CHUNK / RATE)
        mic_mfccs_reduced = mic_mfccs_reduced[:, min_idx:max_idx]

        # Plot Mel-frequency cepstral coefficients (MFCCs) of the microphone audio 繪圖用 之後可刪
        mel_spec = librosa.feature.melspectrogram(y=reduced_audio, sr=RATE)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        plt.figure(figsize=(10, 6))
        librosa.display.specshow(
            mel_spec_db,  sr=RATE, hop_length=CHUNK, x_axis='time', y_axis='mel')
        plt.axis('off')
        # plt.show()
        # Save the image as mel_spectrogram.png
        plt.savefig('mel_spectrogram.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load the image
        # image_path = r"C:\\Users\\chenkayyz\\OneDrive - Garmin\\tryfre\\src\\500_1500.png"
        img = cv2.imread('mel_spectrogram.png')

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert the grayscale image (black to white, white to gray)
        inverted_gray = cv2.bitwise_not(gray)

        # Apply edge detection
        edges = cv2.Canny(inverted_gray, threshold1=100,
                          threshold2=250, apertureSize=3)

        # Use the Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180,
                               threshold=self.threshold_control.value())

        # Initialize counters for slopes
        slope_0_count = 0
        slope_1_count = 0
        slope_minus_1_count = 0

        # Iterate through the detected lines and calculate slopes
        if lines is not None:
            for line in lines:
                r, theta = line[0]
                x0 = r * np.cos(theta)
                y0 = r * np.sin(theta)
                x1 = int(x0 - 1000 * np.sin(theta))
                y1 = int(y0 + 1000 * np.cos(theta))
                x2 = int(x0 + 1000 * np.sin(theta))
                y2 = int(y0 - 1000 * np.cos(theta))
                # 將線的顏色更改為紅色，寬度更改為2像素
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Calculate the slope
                if np.sin(theta) == 0:
                    slope = float('inf')
                else:
                    slope = np.cos(theta) / np.sin(theta)

                # Apply your logic to count slopes
                if abs(slope) > 5:
                    continue
                elif slope == float('inf'):
                    continue
                elif abs(slope) < 0.05:
                    slope_0_count += 1
                elif 0.05 < slope < 5:
                    slope_1_count += 1
                elif -0.05 > slope > -5:
                    slope_minus_1_count += 1

            # Determine which category has the most counts using max()
            max_count_category = max(
                slope_0_count, slope_1_count, slope_minus_1_count)

            if slope_0_count == slope_minus_1_count == slope_1_count == 0:
                print("can not classify")
                self.soundDirection = 'none'
                self.result_display.setText("can not classify")
            elif max_count_category == slope_0_count:
                print("single")
                self.soundDirection = 'single'
                self.result_display.setText("single")

            elif max_count_category == slope_1_count:
                print("up")
                self.soundDirection = 'up'
                self.result_display.setText("up")
            elif max_count_category == slope_minus_1_count:
                print("down")
                self.soundDirection = 'down'
                self.result_display.setText("down")
            else:
                print("No dominant category")
                self.result_display.setText("No dominant category")

            # Display the statistics
            print(f"Slope 0 Count: {slope_0_count}")
            print(f"Slope 1 Count: {slope_1_count}")
            print(f"Slope -1 Count: {slope_minus_1_count}")

            self.single_display.setText(str(slope_0_count))
            self.up_display.setText(str(slope_1_count))
            self.down_display.setText(str(slope_minus_1_count))

            height, width, channel = img.shape
            bytesPerline = channel * width
            # Display the original image with detected lines
            self.qImg = QImage(img, width, height,
                               bytesPerline, QImage.Format_BGR888)
            self.audio_label.setPixmap(QPixmap.fromImage(self.qImg))
            
            self.audio_start = True



if __name__ == '__main__':
    ocv = True
    app = QtWidgets.QApplication(sys.argv)
    main_windows = MainWindows()

    main_windows.show()
    main_windows.closeEvent = main_windows.closeOpenCV
    sys.exit(app.exec_())
