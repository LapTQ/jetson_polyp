import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtGui, QtWidgets


# deal with cmd-line arguments
app = QApplication([])

# GUI
window = QWidget()  # top-level window
window.setWindowTitle('Polyp')
window.setGeometry(100, 100, 280, 80)   # (x, y, w, h)


layout_main = QtWidgets.QVBoxLayout()

layout_option = QtWidgets.QHBoxLayout()
layout_main.addLayout(layout_option)

layout_option_selections = QtWidgets.QGridLayout()
layout_option.addLayout(layout_option_selections)

# selection
SELECTION_MIN_WIDTH = 250
label_model = QtWidgets.QLabel('Chọn mô hình:')
layout_option_selections.addWidget(label_model, 1, 1)

label_thresh = QtWidgets.QLabel('Chọn ngưỡng:')
layout_option_selections.addWidget(label_thresh, 1, 2)

dropdown_model = QtWidgets.QComboBox()
dropdown_model.addItem('Phân vùng 1')
dropdown_model.addItem('Nhận diện 1')
dropdown_model.setMinimumWidth(SELECTION_MIN_WIDTH)
layout_option_selections.addWidget(dropdown_model, 2, 1)

slider_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
slider_thresh.setMinimum(0) # only provides integer ranges
slider_thresh.setMaximum(10)
slider_thresh.setSingleStep(1)
slider_thresh.setValue(5)
slider_thresh.setTickInterval(1)
slider_thresh.setTickPosition(QtWidgets.QSlider.TicksAbove)
slider_thresh.setMinimumWidth(SELECTION_MIN_WIDTH)
def trigger_slider_change(value):
    print(value)
slider_thresh.valueChanged.connect(trigger_slider_change)
layout_option_selections.addWidget(slider_thresh, 2, 2)

# start button
START_MIN_WIDTH = 125
START_MIN_HEIGHT = 40
button_start = QtWidgets.QPushButton('BẮT ĐẦU')
button_start.setToolTip('This is tool tip')
button_start.setMinimumWidth(START_MIN_WIDTH)
button_start.setMinimumHeight(START_MIN_HEIGHT)
layout_option.addWidget(button_start)


# # displayer
# DISPLAYER_MIN_HEIGHT = 500
# displayer = QtWidgets.QLabel() # parent=window
# pixmap = QtGui.QPixmap('/home/tran/Downloads/Lenna.png')
# # displayer.resize(100, 100)
# # displayer.setPixmap(pixmap.scaled(displayer.size(), QtCore.Qt.KeepAspectRatio))
# displayer.setPixmap(pixmap)
# displayer.setScaledContents(False)
# layout_main.addWidget(displayer, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)



window.setLayout(layout_main)
# window.showMaximized()

window.show()
sys.exit(app.exec())
