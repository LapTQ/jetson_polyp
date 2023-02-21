from pathlib import Path
import os
import json
import logging

# "/home/s/Downloads/191102 2.mp4"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t|%(levelname)s\t|%(funcName)s\t|%(lineno)d\t|%(message)s'
)

HERE = Path(__file__).parent

CONFIG_PATH = str(HERE / 'app_config.json')
WEIGHTS_DIR = str(HERE / '../weights')

# logging.debug(f">>>>>> {CONFIG_PATH}, {WEIGHTS_DIR}")


import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5 import QtCore, QtGui, QtWidgets


global start
start = False


def get_input(config):

    global threshold
    global model_type
    global classify_tumor
    global start_app
    model_type = config['model_type']
    threshold = config['threshold']
    classify_tumor = True if config['n_classes'] == 2 else False
    start_app = False


    # deal with cmd-line arguments
    app = QApplication([])

    # GUI
    window = QWidget()  # top-level window
    window.setWindowTitle('Tùy chọn mô hình')
    window.setGeometry(100, 100, 280, 80)   # (x, y, w, h)


    layout_main = QtWidgets.QVBoxLayout()

    layout_option = QtWidgets.QHBoxLayout()
    layout_main.addLayout(layout_option)

    layout_option_selections = QtWidgets.QGridLayout()
    layout_option.addLayout(layout_option_selections)

    # selection
    SELECTION_MIN_WIDTH = 250
    label_model = QtWidgets.QLabel('Chọn loại mô hình:')
    layout_option_selections.addWidget(label_model, 1, 1)

    label_thresh = QtWidgets.QLabel('Chọn ngưỡng:')
    layout_option_selections.addWidget(label_thresh, 1, 2)

    segmentation_label = 'Phân vùng'
    detection_label = 'Phát hiện'
    dropdown_model = QtWidgets.QComboBox()
    dropdown_model.addItem(segmentation_label)
    dropdown_model.addItem(detection_label)
    dropdown_model.setCurrentText(segmentation_label if model_type == 'segmentation' else detection_label)
    dropdown_model.setMinimumWidth(SELECTION_MIN_WIDTH)
    def _trigger_dropdown_model(value):
        global model_type
        global threshold
        logging.debug(f'Triggering Dropdown to value {value}')
        if value == segmentation_label:
            model_type = 'segmentation'
            threshold = config['threshold']
            slider_thresh.setValue(int(threshold * 10))
            logging.debug(f'Segmentation model is selected => restore previous threshold to {threshold}')
        else:
            model_type = 'detection'
            
        logging.info(f'Model type is set to {model_type}')
    dropdown_model.currentTextChanged.connect(_trigger_dropdown_model)
    layout_option_selections.addWidget(dropdown_model, 2, 1)


    slider_thresh = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider_thresh.setMinimum(0) # only provides integer ranges
    slider_thresh.setMaximum(10)
    slider_thresh.setSingleStep(1)
    slider_thresh.setValue(int(config['threshold'] * 10))
    slider_thresh.setTickInterval(1)
    slider_thresh.setTickPosition(QtWidgets.QSlider.TicksAbove)
    slider_thresh.setMinimumWidth(SELECTION_MIN_WIDTH)
    def _trigger_slider_change(value):
        global threshold
        if model_type == 'segmentation':
            threshold = config['threshold']
            slider_thresh.setValue(int(threshold * 10))
        else:
            threshold = value / 10.
            logging.info(f'Threshold is set to {threshold}')
    slider_thresh.valueChanged.connect(_trigger_slider_change)
    layout_option_selections.addWidget(slider_thresh, 2, 2)


    checkbox_classify_tumor = QtWidgets.QCheckBox('Phân loại lành/ác')
    checkbox_classify_tumor.setChecked(classify_tumor)
    def _trigger_checkbox_change(value):
        global classify_tumor
        classify_tumor = False if value == 0 else True
        logging.info(f'Classifying tumor is set to {classify_tumor}')
    checkbox_classify_tumor.stateChanged.connect(_trigger_checkbox_change)
    layout_option.addWidget(checkbox_classify_tumor)
    # layout_option_selections.addWidget(checkbox_classify_tumor, 1, 3)

    # start button
    START_MIN_WIDTH = 125
    START_MIN_HEIGHT = 40
    button_start = QtWidgets.QPushButton('BẮT ĐẦU')
    button_start.setToolTip('This is tool tip')
    button_start.setMinimumWidth(START_MIN_WIDTH)
    button_start.setMinimumHeight(START_MIN_HEIGHT)
    def _trigger_start_button():
        global start_app
        start_app = True
        logging.info('START button clicked')
        input_ = {
            'model_type': model_type,
            'threshold': threshold,
            'classify_tumors': classify_tumor
        }
        update_config(config, input_)
        logging.info(f'Saving new config at {CONFIG_PATH} with {input_}')
        # with open(CONFIG_PATH, 'w') as f:
        #     json.dump(config, f) 
        window.close()
    button_start.clicked.connect(_trigger_start_button)
    layout_option.addWidget(button_start)
    

    window.setLayout(layout_main)
    window.show()
    app.aboutToQuit.connect(lambda config=config: close_gui(config))
    sys.exit(app.exec())


def close_gui(config):
    config['start_app'] = start_app
    logging.info(f'start_app == {start_app}')
    with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f) 



def update_config(config, input_):

    config['start_app'] = True

    config['model_type'] = input_['model_type']
    config['threshold'] = input_['threshold']
    if config['model_type'] == 'segmentation':
        logging.info('Segmentation model is selected.')

        # model path
        config['engine_file_path'] = os.path.join(WEIGHTS_DIR, 'segmentation_onnx', 'repvgg-a1-fulldim.onnx')
        config['engine_type'] = 'onnx'
        # config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'segmentation_trt', 'model.trt')
        # config['engine_type'] = 'trt'

        # set input size
        config['width'] = 384
        config['height'] = 384

        # set preprocess and postprocess mode
        config['preprocess_mode'] = 5
        config['postprocess_mode'] = 2
    
    elif config['model_type'] == 'detection':
        logging.info('Detection model is selected.')

        # model path
        config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'detection_onnx', 'model.onnx')
        config['engine_type'] = 'onnx'
        # config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'detection_trt', 'model.trt')
        # config['engine_type'] = 'trt'
        # config['engine_file_path'] =  os.path.join(WEIGHTS_DIR, 'detection_tftrt_fp16', 'trt_FP16_2')
        # config['engine_type'] = 'tftrt'

        # set input size
        config['width'] = 640
        config['height'] = 640

        # set preprocess_mode
        config['preprocess_mode'] = 6
        config['postprocess_mode'] = 3

        # set class_labels
        config['class_labels'] = ['polyp', 'neo']            
    
    # Opt for classifying tumors or not
    if input_['classify_tumors'] is True:
        config['n_classes'] = 2
    elif input_['classify_tumors'] is False:
        config['n_classes'] = 1
    logging.info(f"Number of classes is set to {config['n_classes']}")


def main():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    get_input(config)


if __name__ == '__main__':
    main()
