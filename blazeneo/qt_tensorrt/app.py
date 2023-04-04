import os
import sys
import time
import cv2
import numpy as np
from queue import Queue
from pathlib import Path
from select_roi import main as select_roi


HERE = Path(__file__).parent

CONFIG_PATH = str(HERE / "app_config.json")

from libs.pipeline import Streamming, Inferencing, Displaying, Config
from libs.utilities import Preprocess, Postprocess

def running(cfg):
    while True:
        if cfg.is_stop():
            break


def main():


    os.system(f'python3 {HERE / "select_model_gui.py"}')
    select_roi()
    
    cfg = Config(CONFIG_PATH)

    preprocess = Preprocess(cfg.width, cfg.height, cfg.preprocess_mode)     # w=384,h=384,preprocess_mode=5
    preprocessing = preprocess.run
    postprocess = Postprocess(cfg.x_start, cfg.y_start, cfg.x_end, cfg.y_end, cfg.height, cfg.width, cfg.n_classes, cfg.threshold, cfg.iou_threshold, cfg.postprocess_mode, cfg.class_labels)    # w=1280, h=720, n_classes=2, threshold=0.6, postprocess_mode=0    # TODO fix  (app_config + Preprocess) theo model
    postprocessing = postprocess.run

    streamming = Streamming(cfg)
    if not streamming.is_opened():
        return
    inferencing = Inferencing(preprocessing, postprocessing, cfg)
    displaying = Displaying(cfg)
    
    
    while True:
        ret, in_image = streamming.read()
        if not ret:
            break
        out_image = inferencing.process(in_image)
        displaying.display(out_image)


if __name__ == "__main__":
    main()