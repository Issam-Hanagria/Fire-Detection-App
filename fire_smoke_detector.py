import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import sys
import torch
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

# Add the YOLOv5 repository to the Python path
yolov5_path = Path('yolov5')
sys.path.append(str(yolov5_path))

from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords 
from yolov5.utils.dataloaders import letterbox
from yolov5.utils.torch_utils import select_device
# Path to your custom YOLOv5 model weights (best.pt)
model_path = Path('model/weights/best_Dalles2.pt')

# Select device
device = select_device('cpu')  # Specify 'cuda' if you have a GPU
print(f"Selected device: {device}")

# Load the YOLOv5 model
model = attempt_load(model_path)  # Remove map_location argument
model.to(device)  # Manually move the model to the specified device

print("Model loaded successfully!")

# # Load your YOLOv5s model
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Define confidence thresholds for fire and smoke classes
fire_conf_threshold = 0.6  # Adjust as needed
smoke_conf_threshold = 0.1  # Adjust as needed

class FireSmokeDetectorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.capture = cv2.VideoCapture(0)  # Use 0 for the default camera
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def initUI(self):
        self.setWindowTitle('Fire and Smoke Detection')
        self.setGeometry(100, 100, 800, 600)
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setGeometry(10, 10, 780, 580)
        self.show()

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert frame color space from RGB to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Resize frame to 640x640
            frame = cv2.resize(frame, (640, 640))

            # Convert frame to a torch tensor
            img = torch.from_numpy(frame).to(device)
            img = img.permute(2, 0, 1).float()  # Change shape to [3, H, W]
            img /= 255.0  # Normalize to [0, 1]
            img = img.unsqueeze(0)  # Add batch dimension [1, 3, H, W]

            # Run the model
            with torch.no_grad():
                results = model(img)[0]

            # Apply NMS (non-maximum suppression)
            results = non_max_suppression(results)

            # Process results
            frame = self.plot_boxes(results, frame, img)
            image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QtGui.QImage.Format_RGB888)
            self.video_label.setPixmap(QtGui.QPixmap.fromImage(image))




    def plot_boxes(self, results, frame, img):
        for det in results:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in det:
                    # Check if the class is fire or smoke and adjust confidence threshold accordingly
                    if int(cls) == 0:  # Fire class index
                        if conf >= fire_conf_threshold:
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            self.draw_box(frame, xyxy, label)
                    elif int(cls) == 1:  # Smoke class index
                        if conf >= smoke_conf_threshold:
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            self.draw_box(frame, xyxy, label)
        return frame

    def draw_box(self, img, xyxy, label=None, color=(255, 0, 0), line_thickness=2):
        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            tf = max(line_thickness - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = FireSmokeDetectorApp()
    sys.exit(app.exec_())