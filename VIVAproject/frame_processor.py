import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input # type: ignore
import onnxruntime as ort
import torch
from torchvision.ops import nms as torch_nms


import sqlite3
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

DATABASE = 'admin.db'

class FrameProcessor:
    def __init__(self, camera, yolo_path, resnet_path, save_dir="cheating_frames"):
        self.camera = camera
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        # Loading YOLO ONNX model
        self.ort_session = ort.InferenceSession(yolo_path)
        self.input_name = self.ort_session.get_inputs()[0].name

        # Loading ResNet model
        self.resnet_model = load_model(resnet_path)

        # Save directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.frame_count = 0
        self.start_time = time.time()

        threading.Thread(target=self._run, daemon=True).start()

    def preprocess_yolo(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess_yolo(self, outputs, frame_shape, conf_threshold=0.25, iou_threshold=0.45):
        predictions = np.squeeze(outputs[0])
        boxes, scores, class_ids = [], [], []

        for det in predictions:
            conf = det[4]
            if conf < conf_threshold:
                continue
            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))
            if class_scores[class_id] < conf_threshold:
                continue

            cx, cy, w, h = det[:4]
            x_scale = frame_shape[1] / 640
            y_scale = frame_shape[0] / 640
            xmin = (cx - w / 2) * x_scale
            ymin = (cy - h / 2) * y_scale
            xmax = (cx + w / 2) * x_scale
            ymax = (cy + h / 2) * y_scale

            boxes.append([xmin, ymin, xmax, ymax])
            scores.append(conf)
            class_ids.append(class_id)

        if not boxes:
            return []

        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        keep_indices = torch_nms(boxes_tensor, scores_tensor, iou_threshold)

        results = []
        for idx in keep_indices:
            i = int(idx)
            xmin, ymin, xmax, ymax = map(int, boxes[i])
            results.append((xmin, ymin, xmax, ymax, scores[i], class_ids[i]))

        return results

    def preprocess_all_crops(self, crops):
        batch = []
        for crop in crops:
            resized = cv2.resize(crop, (224, 224))
            preprocessed = preprocess_input(resized.astype(np.float32))
            batch.append(preprocessed)
        return np.array(batch)

    def _run(self):
        while self.running:
            success, frame = self.camera.read()
            if not success or frame is None:
                continue

            try:
                self.frame_count += 1
                frame_resized = cv2.resize(frame, (640, 640))
                input_tensor = self.preprocess_yolo(frame_resized)
                outputs = self.ort_session.run(None, {self.input_name: input_tensor})
                detections = self.postprocess_yolo(outputs, frame.shape)

                crops = []
                boxes_info = []

                for xmin, ymin, xmax, ymax, conf, class_id in detections:
                    if class_id != 0:
                        continue
                    crop = frame[ymin:ymax, xmin:xmax]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    boxes_info.append((xmin, ymin, xmax, ymax))

                if crops:
                    batched_crops = self.preprocess_all_crops(crops)
                    predictions = self.resnet_model.predict(batched_crops)

                    for i, pred in enumerate(predictions):
                        label = "Cheating" if pred[0] > 0.5 else "Not Cheating"
                        confidence = float(pred[0])
                        xmin, ymin, xmax, ymax = boxes_info[i]
                        color = (0, 0, 255) if label == "Cheating" else (0, 255, 0)

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, f"{label} ({confidence:.2f})", (xmin, ymin - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        if label == "Cheating":
                            timestamp = datetime.now().strftime("%d-%m-%Y__%I-%M-%S-%p-%f")
                            filename = os.path.join(self.save_dir, f"cheating_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)

                            admin_email = get_admin_email()
                            if admin_email:
                                message = f"Cheating detected at {timestamp}.\nImage saved as {filename}."
                                send_alert_notification(admin_email, message)

                elapsed_time = time.time() - self.start_time
                fps = self.frame_count / elapsed_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)

                with self.lock:
                    self.latest_frame = frame

            except Exception as e:
                print(f"[ERROR] Frame processing: {e}")

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn




def get_admin_email():
    conn = get_db_connection()
    admin = conn.execute("SELECT email FROM admin LIMIT 1").fetchone()
    conn.close()
    return admin["email"] if admin else None


def send_alert_notification(recipient_email, message):
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

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