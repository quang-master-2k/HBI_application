import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Text Field Example")
        self.layout = QVBoxLayout()
        self.label = QLabel("Enter text:")
        self.text_field = QLineEdit()
        self.button = QPushButton("Get Text")
        self.button.clicked.connect(self.get_text)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text_field)
        self.layout.addWidget(self.button)
        self.setLayout(self.layout)

    def get_text(self):
        text = self.text_field.text()
        print("Text entered:", text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
