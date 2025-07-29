from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, Query, status, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, StreamingResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware 
from starlette.status import HTTP_302_FOUND
import threading
import torchvision.ops as ops

import onnxruntime as ort

from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from src.schemas import Offer

from threaded_camera import ThreadedCamera
from frame_processor import FrameProcessor

from werkzeug.security import generate_password_hash, check_password_hash
import uvicorn
import librosa
from keras_yamnet import params
from keras_yamnet.yamnet import YAMNet, class_names
from keras_yamnet.preprocessing import preprocess_input as yamnet_preprocessing

import pyaudio
import zipfile
import io
import cv2
import threading
import asyncio
import os
import platform
import time
import torch
import math
import re
from bleak import BleakScanner
from datetime import datetime
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import numpy as np
import sqlite3
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()



app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("FLASK_SECRET_KEY"))
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


room_radius = None 
running_detection = False
ble_devices = []
wifi_networks = []
history_ble = []
history_wifi = []

# Audio config
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = params.SAMPLE_RATE
CHUNK = int(0.975 * RATE)
MIC = 1
audio = pyaudio.PyAudio()
stream = None
latest_result = "None"
# Model setup
yamnet_classes = class_names(r"E:\my_files\final_year_project\flasktofastapi\keras_yamnet\yamnet_class_map.csv")
plt_classes = [0, 2, 6, 12, 42, 44, 494]
plt_classes_lab = yamnet_classes[plt_classes]
model = YAMNet(weights=r"E:\my_files\final_year_project\flasktofastapi\keras_yamnet\yamnet.h5")



DATABASE = 'admin.db'


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VideoTransformTrack(VideoStreamTrack):
    def __init__(self, rtsp_link):
        super().__init__()
        self.processor = FrameProcessor(
            ThreadedCamera(rtsp_link),
            yolo_path=os.getenv("ONNX_PATH"),
            resnet_path=os.getenv("RESNET_MODEL_PATH")
        )

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        frame = self.processor.get_latest_frame()
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        return new_frame


def calculate_distance(rssi, tx_power=-69, n=2):
    return round(10 ** ((tx_power - rssi) / (10 * n)), 2)


def wifi_distance(signal_percent):
    if signal_percent >= 90:
        return 1
    elif signal_percent >= 75:
        return 2
    elif signal_percent >= 50:
        return 4
    elif signal_percent >= 30:
        return 6
    else:
        return 8


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            phone TEXT,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def get_admin_email():
    conn = get_db_connection()
    admin = conn.execute("SELECT email FROM admin LIMIT 1").fetchone()
    conn.close()
    return admin["email"] if admin else None


def send_alert_notification(recipient_email, message):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = 587
    smtp_user = os.getenv("SMTP_USER3")
    smtp_password = os.getenv("SMTP_PASSWORD3")

    msg = MIMEText(message)
    msg['Subject'] = 'Alert Notification'
    msg['From'] = smtp_user
    msg['To'] = recipient_email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        print("Alert sent to", recipient_email)
    except Exception as e:
        print("Failed to send alert:", e)


# Reuse the existing scan and processing threads
def scan_ble_devices():
    asyncio.run(ble_scan())


def scan_wifi_networks():
    global wifi_networks, history_wifi

    while running_detection:
        if platform.system() == "Windows":
            output = os.popen('netsh wlan show networks mode=bssid').read()
        else:
            output = os.popen('nmcli -f SSID,BSSID,SIGNAL dev wifi').read()

        timestamp = datetime.now().strftime("%d-%m-%Y__%I:%M:%S %p")
        raw_networks = parse_wifi_output(output)

        unique_ids = set()
        filtered_networks = []

        for network in raw_networks:
            network['timestamp'] = timestamp
            uid = (network.get('ssid'), network.get('bssid'))

            if uid not in unique_ids:
                unique_ids.add(uid)
                filtered_networks.append(network)

                if uid not in [(n['ssid'], n['bssid']) for n in history_wifi]:
                    history_wifi.append(network)
                    admin_email = get_admin_email()
                    if admin_email:
                        message = f"New Wi-Fi network detected:\nSSID: {network['ssid']}\nBSSID: {network['bssid']}\nSignal: {network['signal']}\nEstimated Distance: {network['distance']}m\nTime: {timestamp}"
                        send_alert_notification(admin_email, message)

        wifi_networks = filtered_networks
        time.sleep(5)


async def ble_scan():
    global ble_devices, history_ble
    while running_detection:
        scanner = BleakScanner()
        devices = await scanner.discover(timeout=5)
        ble_devices = []

        for device in devices:
            name = device.name or "Unknown"
            address = device.address
            rssi = device.rssi
            distance = calculate_distance(rssi)
            timestamp = datetime.now().strftime("%d-%m-%Y__%I:%M:%S %p")
            signal_type = "Bluetooth"

            device_tuple = (name, address, distance, rssi, signal_type, timestamp)
            ble_devices.append(device_tuple)

            if all(d[1] != address for d in history_ble):
                history_ble.append(device_tuple)
                admin_email = get_admin_email()
                if admin_email:
                    message = f"New Bluetooth device detected:\nName: {name}\nAddress: {address}\nDistance: {distance}m\nRSSI: {rssi} dBm\nTime: {timestamp}"
                    send_alert_notification(admin_email, message)

        await asyncio.sleep(5)


def parse_wifi_output(output):
    networks = []
    lines = output.splitlines()
    ssid = None
    bssid = None

    if platform.system() == "Windows":
        for line in lines:
            line = line.strip()
            if line.startswith("SSID"):
                match = re.match(r"SSID\s+\d+\s+:\s+(.*)", line)
                if match:
                    ssid = match.group(1)
            elif line.startswith("BSSID"):
                bssid_match = re.match(r"BSSID\s+\d+\s+:\s+(.*)", line)
                if bssid_match and ssid:
                    bssid = bssid_match.group(1)
            elif line.startswith("Signal") and ssid and bssid:
                signal_match = re.match(r"Signal\s+:\s+(\d+)%", line)
                if signal_match:
                    signal = int(signal_match.group(1))
                    distance = wifi_distance(signal)
                    networks.append({
                        'ssid': ssid,
                        'bssid': bssid,
                        'signal': f"{signal}%",
                        'distance': distance
                    })
    else:
        lines = output.strip().split('\n')[1:]
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                signal = int(parts[-1])
                bssid = parts[-2]
                ssid = ' '.join(parts[:-2])
                distance = wifi_distance(signal)
                networks.append({
                    'ssid': ssid,
                    'bssid': bssid,
                    'signal': f"{signal}%",
                    'distance': distance
                })

    return networks


def detect_audio_loop():
    global latest_result, stream,running_detection

    stream = audio.open(format=FORMAT,
                        input_device_index=MIC,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    while running_detection:
        try:
            data = yamnet_preprocessing(np.frombuffer(stream.read(CHUNK), dtype=np.float32), RATE)
            prediction = model.predict(np.expand_dims(data, 0), verbose=0)[0]
            selected_predictions = prediction[plt_classes]
            top_class_index = np.argmax(selected_predictions)
            top_class_label = plt_classes_lab[top_class_index]
            top_class_score = selected_predictions[top_class_index]
            

            latest_result = f"Detected: {top_class_label} ({top_class_score:.2f})"
            admin_email = get_admin_email()
            if top_class_label.lower() == "speech" and admin_email:
                timestamp = datetime.now().strftime("%d-%m-%Y__%I:%M:%S %p")
                message = f"Speech detected at {timestamp}"
                send_alert_notification(admin_email, message)
        except Exception as e:
            latest_result = f"Error: {str(e)}"
            break

    stream.stop_stream()
    stream.close()

    



def start_detection():
    global running_detection
    running_detection = True
    
    threading.Thread(target=scan_ble_devices, daemon=True).start()
    threading.Thread(target=scan_wifi_networks, daemon=True).start()
    threading.Thread(target=detect_audio_loop, daemon=True).start()
    


def stop_detection():
    global running_detection
    running_detection = False
    return {"status": "stopped"}




# Init DB on app startup
@app.on_event("startup")
async def startup_event():
    init_db()


# Endpoint converted from Flask's `/start`

@app.get("/start")
async def start_detection_endpoint(width: float = Query(0), length: float = Query(0)):
    global running_detection, room_radius

    if width > 0  and length > 0:
        room_radius = round(math.sqrt(width**2 + length**2) / 2, 2)
    else:
        room_radius = None
        
    if not running_detection:
            start_detection()


    return JSONResponse(content={
        'status': 'Detection started',
        'radius': room_radius  # This will be null in JSON if not calculated
    })



@app.get("/stop")
async def stop():
    if running_detection:
        stop_detection()
    return {"status": "Detection stopped"}


@app.get("/ble_devices")
async def get_ble():
    return {"devices": ble_devices}


@app.get("/wifi_networks")
async def get_wifi():
    return {"networks": wifi_networks}



@app.get("/get_result")
async def get_result():
    global latest_result
    return latest_result


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request):
    return templates.TemplateResponse("result.html", {
        "request": request,
        "ble": ble_devices,             # Should be a list of 6-tuples
        "wifi": history_wifi,           # Should be a list of dicts
        "radius": room_radius           # float
    })


@app.get("/download")
async def download_devices():
    unique_data = {device[1]: device for device in history_ble}
    def generate():
        yield "Name,Address,Distance (m),RSSI (dBm),Signal Type,Timestamp\n"
        for name, address, distance, rssi, signal_type, timestamp in unique_data.values():
            yield f"{name},{address},{distance:.2f},{rssi},{signal_type},{timestamp}\n"
    return StreamingResponse(generate(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=detected_devices.csv"})


@app.get("/download_cheating_frames")
async def download_cheating_frames():
    folder = "cheating_frames"
    if not os.path.exists(folder):
        return PlainTextResponse("No frames available", status_code=404)

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            zipf.write(filepath, arcname=filename)
    zip_io.seek(0)
    return Response(zip_io.read(), media_type='application/zip', headers={"Content-Disposition": "attachment; filename=cheating_frames.zip"})

@app.get("/back", response_class=HTMLResponse)
async def backbtn(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
@app.post("/signup", response_class=HTMLResponse)
async def signup(request: Request, username: str = Form(None), email: str = Form(None),
                 phone: str = Form(None), password: str = Form(None)):
    if request.method == "POST": 
        conn = get_db_connection()
        existing_user = conn.execute("SELECT * FROM admin WHERE username = ?", (username,)).fetchone()
        if existing_user:
            return templates.TemplateResponse("signup.html", {
                "request": request, 
                "message": "Username already taken."
            })
        
        hashed_password = generate_password_hash(password)
        conn.execute(
            "INSERT INTO admin (username, email, phone, password) VALUES (?, ?, ?, ?)",
            (username, email, phone, hashed_password)
        )
        conn.commit()
        conn.close()
        return RedirectResponse(url="/signin", status_code=HTTP_302_FOUND)

    return templates.TemplateResponse("signup.html", {"request": request})


@app.get("/signin", response_class=HTMLResponse)
@app.post("/signin", response_class=HTMLResponse)
async def signin(request: Request, username: str = Form(None), password: str = Form(None)):
    if request.method == "POST":
        conn = get_db_connection()
        admin = conn.execute('SELECT * FROM admin WHERE username = ?', (username,)).fetchone()
        conn.close()
        if admin and check_password_hash(admin[4], password):
            request.session['admin_id'] = admin[0]
            request.session['admin_username'] = admin[1]
            return RedirectResponse(url="/", status_code=HTTP_302_FOUND)
        else:
            return templates.TemplateResponse("signin.html", {"request": request, "message": "Invalid credentials."})
    return templates.TemplateResponse("signin.html", {"request": request})


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()  # This clears 'admin_id'
    return RedirectResponse(url="/signin", status_code=HTTP_302_FOUND)


async def login_required(request: Request):
    if not request.session.get("admin_id"):
       return RedirectResponse(url="/signin", status_code=HTTP_302_FOUND)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not request.session.get("admin_id"):
        return RedirectResponse(url="/signin", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/trigger_alert")
async def trigger_alert(request: Request):
    admin_id = request.session.get("admin_id")
    conn = get_db_connection()
    admin = conn.execute('SELECT * FROM admin WHERE id = ?', (admin_id,)).fetchone()
    conn.close()
    if admin:
        send_alert_notification(admin[2], "This is an alert notification.")
    return RedirectResponse(url="/", status_code=HTTP_302_FOUND)


@app.post("/offer")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    print("calling Video Capture Class!")
    pc.addTrack(
        VideoTransformTrack(rtsp_link=os.getenv("RTSP_URL")))

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(answer)
    print("Before Returning Offer!")

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

pcs = set()
args = ''

async def shutdown():
  # Close peer connections
  coros = [pc.close() for pc in pcs]
  await asyncio.gather(*coros)
  pcs.clear()

app.add_event_handler("shutdown", shutdown)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
