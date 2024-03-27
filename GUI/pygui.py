import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGridLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSizePolicy

class MessageWindow(QWidget):
    def __init__(self, message):
        super().__init__()
        self.setWindowTitle("Message")
        layout = QVBoxLayout()
        label = QLabel(message)
        layout.addWidget(label)
        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 300)  # Adjust size as needed
        self.move(1500, 1000)  # Adjust position as needed

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Grid Buttons Example")
        self.setGeometry(1000, 1000, 1000, 1000)  # Set initial window size
        self.create_grid_buttons()

    def create_grid_buttons(self):
        layout = QGridLayout()

        # Assuming a 2x4 grid for 8 pieces below the label
        positions = [(i, j) for i in range(1, 3) for j in range(4)]  # Adjusted range for rows to accommodate the label
        for position, name in zip(positions, range(1, 9)):
            button = QPushButton(f"Button {name}")
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            button.clicked.connect(lambda checked, name=name: self.update_info_label(name))
            layout.addWidget(button, *position)

        self.setLayout(layout)

    def update_info_label(self, button_name):
        msg = f"Button {button_name} clicked"
        self.MessageWindow = MessageWindow(msg)
        self.MessageWindow.show()
    
        

class MouseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mouse")
        self.setGeometry(100, 100, 400, 400)  # Adjust size as needed
        self.initUI()
    
    def initUI(self):
        self.label = QLabel(self)
        self.label.setGeometry(self.rect())  # Make the label fill the entire window
        self.defaultPixmap = QPixmap('mouse.png')
        self.leftclickedPixmap = QPixmap('mouthleftclick.png')
        self.rightclickedPixmap = QPixmap('mouthrightclick.png')
        self.updatePixmap(self.defaultPixmap)
        
    def updatePixmap(self, pixmap):
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)
        
    def mousePressEvent(self, event):
        # Check if the click is on the left part of the window
        if event.button() == Qt.LeftButton and event.pos().x() < self.width() / 2:
            self.updatePixmap(self.leftclickedPixmap)
        elif event.button() == Qt.RightButton and event.pos().x() > self.width() / 2:
            self.updatePixmap(self.rightclickedPixmap)
    
    def mouseReleaseEvent(self, event):
        # Revert to the default image when the mouse button is released
        self.updatePixmap(self.defaultPixmap)
        
    def resizeEvent(self, event):
        # Update the pixmap scaling when the window is resized
        super().resizeEvent(event)
        self.updatePixmap(self.label.pixmap())

if __name__ == "__main__":  
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    image_window = MouseWindow()
    image_window.show()
    sys.exit(app.exec_())
