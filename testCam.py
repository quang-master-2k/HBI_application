import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QRect

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(1178, 824)
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")

        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QRect(20, 90, 722, 482))
        self.graphicsView.setObjectName("graphicsView")
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.timer = QTimer(self.graphicsView)
        self.timer.timeout.connect(self.update_frame)
        self.start_webcam()


    def start_webcam(self):
        self.cap = cv2.VideoCapture(2)
        self.timer.start(30)  # Update frame every 30 milliseconds

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            image = QImage(
                frame_rgb.data, w, h, ch * w, QImage.Format_RGB888
            )
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(720, 480)
            self.scene.clear()
            self.scene.addPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
