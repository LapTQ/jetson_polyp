import numpy as np
import cv2
import json
from pathlib import Path

HERE = Path(__file__).parent

CONFIG_PATH = str(HERE / 'app_config.json')


class Displaying:
    """
    Class that continously display frames with a dedicated thread
    """
    def __init__(self, x_start, y_start, x_end, y_end, name="display video main thread"):
        self.window_name = name
        self.__setupWindow()

        self.cropping = False
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None
        self.current_x = 0
        self.current_y = 0
        self.w = None
        self.h = None

    def __mouse_crop(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1, self.y1, self.x2, self.y2 = x, y, x, y 
            self.cropping = True
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping == True:
                self.x2, self.y2 = x, y 

        elif event == cv2.EVENT_LBUTTONUP:
            self.x2, self.y2 = x, y 
            self.cropping = False 
        
        if self.cropping:
            self.current_x = x
            self.current_y = y

            if self.x1 is not None and self.y1 is not None and self.x2 is not None and self.y2 is not None:
                if abs(self.x2 - self.x1) > 10 and abs(self.y2 - self.y1) > 10:
                    self.x_start = min(self.x1, self.x2)
                    self.x_end = max(self.x1, self.x2)
                    self.y_start = min(self.y1, self.y2)
                    self.y_end = max(self.y1, self.y2)

    def __setupWindow(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(self.window_name, self.__mouse_crop)

    def run(self, image):
        self.__draw_crop_box(image)
        cv2.rectangle(image, (48, 0), (928, 43), color=(0, 180, 0), thickness=-1)
        cv2.putText(image, 'SELECT REGION OF INTEREST [press <ESC> to submit]', (50, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
        cv2.imshow(self.window_name, image)

    def __draw_crop_box(self, image):
        self.h, self.w = image.shape[:2]
        self.x_start = max(min(self.x_start, self.w), 0)
        self.y_start = max(min(self.y_start, self.h), 0)
        self.x_end = max(min(self.x_end, self.w), 0)
        self.y_end = max(min(self.y_end, self.h), 0)
        self.current_x = max(min(self.current_x, self.w), 0)
        self.current_y = max(min(self.current_y, self.h), 0)

        cv2.rectangle(image, (self.x_start, self.y_start), (self.x_end, self.y_end), (0, 0, 192), 2)
        if self.cropping:
            cv2.line(image, (self.current_x, 0), (self.current_x, self.h), (192, 192, 0), 1)
            cv2.line(image, (0, self.current_y), (self.w, self.current_y), (192, 192, 0), 1)

    def get_coordinates(self):
        return self.x_start, self.y_start, self.x_end, self.y_end

    def destroy(self):
        cv2.destroyAllWindows()



def main():
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if not config['start_app']:
        exit(0)
    
    if config['source'] == '0':
        config['source'] = 0
        
    cap = cv2.VideoCapture(config['source'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['camera_height'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['camera_width'])
    
    x_start = config["crop_coordinates"]["x_start"]
    y_start = config["crop_coordinates"]["y_start"]
    x_end = config["crop_coordinates"]["x_end"]
    y_end = config["crop_coordinates"]["y_end"]

    displaying = Displaying(x_start, y_start, x_end, y_end)

    while(True):

        if not cap.isOpened():
            print("camera has not been connected")
            break
        # Capture frame-by-frame
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (config['camera_width'], config['camera_height']))
        displaying.run(frame)
        key = cv2.waitKey(50)
        if key == ord('q') or key == 27:
            break
        x_start, y_start, x_end, y_end = displaying.get_coordinates()

    config["crop_coordinates"]["x_start"] = x_start
    config["crop_coordinates"]["y_start"] = y_start
    config["crop_coordinates"]["x_end"] = x_end
    config["crop_coordinates"]["y_end"] = y_end
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

    # When everything done, release the capture
    cap.release()
    displaying.destroy()



if __name__ == "__main__":
    main()