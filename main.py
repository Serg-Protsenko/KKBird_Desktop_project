import os
import random
import threading
import time
from tkinter import *
from tkinter import messagebox
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from playsound import playsound

import settings
import platform
import ctypes
import logging
from tkinter import filedialog as fd

logging.basicConfig(   # config INFO level for add DATA to log file
    level=logging.INFO,
    filename='logfile.log',
    filemode='a',
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)

SYSTEM = platform.system().lower()  # only for Windows keeps system awake
if SYSTEM == "windows":
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)
    # print("Always On")  # for testing

bird_detect_counter = 0
detect_count = 0
path_to_sounds = r'birds_sounds'
proto_txt_Path = os.path.join(os.getcwd(), "MobileNetSSD_deploy.prototxt.txt")
model_Path = os.path.join(os.getcwd(), "MobileNetSSD_deploy.caffemodel")


# Random play sound from one of bird of prey
def play_random_sound():
    while True:
        # print('Thread is running')  # for testing
        global detect_count
        # print(f'Detect counter is {detect_count}')  # for testing
        if detect_count >= 1:
            detect_count = 0
            # print('Sleeping 3 sec...', end='\n\n')  # for testing
            time.sleep(3)
            if detect_count > 1:
                random_sound_select = random.choice(os.listdir(path_to_sounds))
                path_random_sound = os.path.join(path_to_sounds, random_sound_select)
                playsound(path_random_sound, block=False)
                # print('Random sound: ', random_sound_select)  # for testing
                global bird_detect_counter
                bird_detect_counter += 1
                logging.info(f'Detect counter: {bird_detect_counter}')  # Add info about detection to log file
                # print('Sleeping 10 sec...', end='\n\n')  # for testing
                time.sleep(10)  # wait 10 second when a bird go away
                detect_count = 0
        time.sleep(1)


play_random_sound_thread = threading.Thread(target=play_random_sound)
play_random_sound_thread.daemon = True

play_random_sound_thread.start()


def start_video():
    settings.start_video = True
    settings.start_processing = True
    show_frame()
    logging.info('Start detection')  # Add info about start detection to log file


def stop_video():
    settings.start_video = False
    settings.start_processing = False
    video_label.config(image=main_image)
    logging.info('Stop detection')  # Add info about stop detection to log file


def disable_button(button):
    button.config(state=DISABLED)


def normal_button(button):
    button.config(state=NORMAL)


def open_log():
    filetypes = (('LOG files', '*.LOG'), ('Text files', '*.txt'))
    file_name = fd.askopenfilename(title="Open file", filetypes=filetypes)
    try:
        with open(file_name, 'r') as f:
            file_text = f.read()
    except FileNotFoundError:  # Use it when user is closing window for opening log file
        pass

    if file_name:
        root.wm_attributes('-topmost', False)  # Turn off that root window to be adobe all apps/windows
        window = Toplevel(root)  # Creat daughter window
        window.title("Log File")
        win_text = Text(window)
        win_text.pack(side=LEFT)

        scroll = Scrollbar(window, command=win_text.yview)
        scroll.pack(side=RIGHT, fill=Y)
        win_text.config(yscrollcommand=scroll.set)

        win_text.insert(1.0, file_text)
        # window.grab_set()  # Set log window works priority than root window
        window.focus_set()
        window.wait_window()


def detector(detect):
    global bird_detect_counter
    var.set(f'Detect counter: {bird_detect_counter}')  # Renamed Label from "Bird detection:" to "Detect counter:"
    # return detect


# Detect all connected webcams
def available_cams():
    cams = []
    for i in range(10):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cam.isOpened():
            cams.append(i)
    return cams


# A function that handles events when you choose webcam in list of webcams (Combobox)
def chose_list(event):
    choose_cam = int(list_cams.get())
    global cap
    cap = cv2.VideoCapture(choose_cam, cv2.CAP_DSHOW)


# Create list of available USB cams
valid_cams = available_cams()


def show_frame():
    if not settings.start_video:
        return None
    _, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (800, 600))

    if settings.start_processing:
        frame = process_frame(frame)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)
    video_label.after(10, show_frame)


def process_frame(img):
    # grab the frame dimensions and convert it to a blob
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (350, 350)), 0.007843, (350, 350), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:  # Increased confidence from 0.2 to 0.5 (20% to 50%) in order to reduce false detection
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            if CLASSES[idx] in IGNORE:  # Ignore all classes except class "Bird"
                detector(False)
                continue
            else:
                detector(True)
                global detect_count
                detect_count += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                                         confidence * 100)
            cv2.rectangle(img, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(img, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    return img


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
           "train", "tvmonitor"]

IGNORE = ["background", "aeroplane", "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
          "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(proto_txt_Path, model_Path)  # "MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel"


root = Tk()
root.title('Kind Kind Bird')
root.iconbitmap('scarecrow-icon.ico')
root.geometry('800x640+250+50')
root.resizable(False, False)
root.config(bg='#474e68')

# Main image
load_image = Image.open("scarecrow-main-picture.jpeg")
resized_img = load_image.resize((800, 600))
main_image = ImageTk.PhotoImage(resized_img)

# Create video block
# Create video frame
video_frame = Frame(root, background='#474e68')  # #21242D
video_frame.pack()

# Create video label
video_label = Label(video_frame, image=main_image)  # , text="Press Start Video"
video_label.pack(fill=BOTH, expand=True, side=TOP)

# Create button block

# ttk_style = ttk.Style()
# ttk_style.theme_use('clam')
# ttk_style.configure()

# Create button frame
buttons_frame = Frame(root, height=40, background='#474e68')  # #474e68
buttons_frame.pack(fill=X, side=TOP)

# Create list of webcams
list_cams_label = Label(buttons_frame, text='Choose webcam:', width=14)
list_cams_label.grid(column=0, row=0, padx=17, pady=8)
list_cams = ttk.Combobox(buttons_frame, values=valid_cams, width=14)
list_cams['state'] = 'readonly'  # Fix bug -> It is possible to enter characters in the dropdown field
list_cams.current(0)
list_cams.bind("<<ComboboxSelected>>", chose_list)
list_cams.grid(column=1, row=0, padx=17, pady=8)

# Create start/stop buttons
btn_start_video = Button(buttons_frame, text="Start Video", width=12,
                         command=lambda: [start_video(), disable_button(btn_start_video),
                                          normal_button(btn_stop_video)])
btn_start_video.grid(column=2, row=0, padx=17, pady=8)

btn_stop_video = Button(buttons_frame, text="Stop Video", state=DISABLED, width=12,
                        command=lambda: [stop_video(), disable_button(btn_stop_video),
                                         normal_button(btn_start_video)])
btn_stop_video.grid(column=3, row=0, padx=17, pady=8)


# Detect block
var = StringVar()
detect_label = Label(buttons_frame, textvariable=var, width=14)
detect_label.grid(column=4, row=0, padx=17, pady=8)

# Create button Open log file
btn_open_log = Button(buttons_frame, text="Open Log File", command=open_log, width=12)
btn_open_log.grid(column=5, row=0, padx=17, pady=8)


detector(bird_detect_counter)

# Capture from camera
choose_camera = int(list_cams.get())
try:
    global cap
    cap = cv2.VideoCapture(choose_camera, cv2.CAP_DSHOW)
except Exception:
    messagebox.showerror('Error', 'USB camera access error!')


root.wm_attributes('-topmost', True)  # Set adobe all apps/windows
root.mainloop()
