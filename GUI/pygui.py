import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Grid Buttons Example")
        self.setGeometry(100, 100, 400, 400)  # Set initial window size
        self.create_grid_buttons()

    def create_grid_buttons(self):
        layout = QGridLayout()

        # Assuming a 2x4 grid for 8 pieces
        positions = [(i, j) for i in range(2) for j in range(4)]
        for position, name in zip(positions, range(1, 9)):
            button = QPushButton(f"Button {name}")
            layout.addWidget(button, *position)

        self.setLayout(layout)

class ImageWindow(QWidget):
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
    image_window = ImageWindow()
    image_window.show()
    sys.exit(app.exec_())