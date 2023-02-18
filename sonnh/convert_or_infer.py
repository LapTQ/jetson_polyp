
import onnx
import onnxruntime as ort
import numpy as np
import os
import cv2
import time

import argparse




MODEL_PATH = '/home/s/GUI/sonnh/20230213/savedmodel'
ONNX = False
CONVERT = True
OPSET = 12
CLASS_NUM = 2
BATCH_SIZE = 1
PRECISION = 'FP16'
HEIGHT = 512
OUT = '/'.join(MODEL_PATH.split('/')[:-1] + ['trt' + '_' + PRECISION if not ONNX else 'model.onnx']) 
VIDEO = '/home/s/Downloads/191102 2.mp4' # '/home/s/Downloads/Colon Polyp 191024 1.mp4' # 0 #'/home/s/Downloads/Colon.mp4'
SAVE_VIDEO = False
THRESHOLD = 0.4
LABELS = ['polyp', 'neo']

def resize(size, image):
    h, w, c = image.shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    h = int(h * scale)
    w = int(w * scale)
    padimg = np.zeros((size, size, c), image.dtype)
    padimg[:h, :w] = cv2.resize(image, (w, h))
    return padimg


def rescale_boxes(size, im_shape, boxes):
    w, h = im_shape
    scale_w = size / w
    scale_h = size / h
    scale = min(scale_w, scale_h)
    new_anns = []
    for box in boxes:
        xmin, ymin, xmax, ymax = [int(p/scale) for p in box]
        new_anns.append([xmin, ymin, xmax, ymax])
    return np.array(new_anns)


def draw_border(img, box, color):
    xmin, ymin, xmax, ymax = box
    point1, point2 = (xmin, ymin), (xmin, ymax)
    point3, point4 = (xmax, ymin), (xmax, ymax), 
    line_length = min(xmax-xmin, ymax-ymin)//5

    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    x4, y4 = point4    

    cv2.circle(img, (x1, y1), 3, color, -1)    #-- top_left
    cv2.circle(img, (x2, y2), 3, color, -1)    #-- bottom-left
    cv2.circle(img, (x3, y3), 3, color, -1)    #-- top-right
    cv2.circle(img, (x4, y4), 3, color, -1)    #-- bottom-right

    cv2.line(img, (x1, y1), (x1 , y1 + line_length), color, 2)  #-- top-left
    cv2.line(img, (x1, y1), (x1 + line_length , y1), color, 2)

    cv2.line(img, (x2, y2), (x2 , y2 - line_length), color, 2)  #-- bottom-left
    cv2.line(img, (x2, y2), (x2 + line_length , y2), color, 2)

    cv2.line(img, (x3, y3), (x3 - line_length, y3), color, 2)  #-- top-right
    cv2.line(img, (x3, y3), (x3, y3 + line_length), color, 2)

    cv2.line(img, (x4, y4), (x4 , y4 - line_length), color, 2)  #-- bottom-right
    cv2.line(img, (x4, y4), (x4 - line_length , y4), color, 2)

    return img


def convert():

    if ONNX:
        os.system(f'python3 -m tf2onnx.convert --saved-model {MODEL_PATH} --opset {OPSET} --output {OUT}')
    
    else:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.python.ops.numpy_ops import np_config
        from tensorflow.python.compiler.tensorrt import trt_convert as trt

        tf_model = tf.keras.models.load_model(MODEL_PATH)
        print(tf_model.summary())

        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=MODEL_PATH,
            precision_mode=eval(f'trt.TrtPrecisionMode.{PRECISION}')
        )

        trt_func = converter.convert()
        converter.summary()


        def input_fn():
           x = np.zeros((BATCH_SIZE, HEIGHT, HEIGHT, 3)).astype('float32')
           yield [x]
         
        converter.build(input_fn=input_fn)
        converter.save(output_saved_model_dir=OUT)
    
    print('[INFO] saved in', OUT)



def infer():


    if ONNX:
        model = onnx.load(MODEL_PATH)
        onnx.checker.check_model(model)

        session = ort.InferenceSession(
            MODEL_PATH,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        io_binding = session.io_binding()
    
    
    else:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.python.ops.numpy_ops import np_config
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        
        # np_config.enable_numpy_behavior()
        print(tf.config.list_physical_devices('GPU'))

        model = tf.saved_model.load(OUT)
        func = model.signatures['serving_default']
        output_tensorname = list(func.structured_outputs.keys())[0]


    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) #  1920 1366
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768) # 1080 768
    cv2.namedWindow('polyp', cv2.WINDOW_NORMAL)
    if SAVE_VIDEO:
        out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (740, 768))

    fps = 0

    while True:
        m1 = time.time()
        success, frame = cap.read()
        if not success:
            break

        frame = frame[:,:740,:]    # [:,:740,:]

        input_ = frame.copy()
        input_ = cv2.cvtColor(input_, cv2.COLOR_BGR2RGB)
        input_ = resize(HEIGHT, input_) # cv2.resize(input_, (HEIGHT, HEIGHT)) # resize
        input_ = np.float32(input_) / 255.
        #print(input_.shape)

        m2 = time.time()
        
        if ONNX:
            input_ = np.expand_dims(input_, axis=0)
            #input_ = ort.OrtValue.ortvalue_from_numpy(input_, 'cuda', '0')
            io_binding.bind_cpu_input('input_1', input_)
            io_binding.bind_output('model')
            session.run_with_iobinding(io_binding)
            output = io_binding.copy_outputs_to_cpu()[0]
        else:
            input_ = tf.expand_dims(input_, axis=0)
            output = func(input_)[output_tensorname]

        m3 = time.time()

        #print(output.shape)
        output = np.squeeze(output)


        boxes, scores, class_ids = output[..., :4], output[..., 4], output[..., 5].astype('int32')
        h, w = frame.shape[:2]
        print(h, w)
        boxes = boxes * HEIGHT
        boxes = rescale_boxes(HEIGHT, (w, h), boxes)
        idxs = np.where(scores > THRESHOLD)
        boxes, scores, class_ids = boxes[idxs], scores[idxs], class_ids[idxs]
        class_names = [LABELS[c] for c in class_ids]

        for box, score, name in zip(boxes, scores, class_names):
            xmin, ymin, xmax, ymax = box
            #score = '{:.4f}'.format(score)
            label = '{:.4f}'.format(score)
            # label = '-'.join([name, score])
            ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        #     cv2.rectangle(frame, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (255, 255, 255), -1)
        #     cv2.putText(frame, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if CLASS_NUM == 2:
                color = (0, 0, 255) if name == 'neo' else (0, 255, 0)
            else:
                color = (0, 255, 0)
            
            frame = draw_border(frame, box, color)
            cv2.rectangle(frame, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), (255, 255, 255), -1)
            cv2.putText(frame, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        m4 = time.time()

        #print('Read:', m2-m1)
        #print('Infer:', m3-m2)
        #print('Draw:', m4-m3)
        fps = 0.75 * fps + 0.25 * 1/(m4-m1)
        print('FPS:', fps)

        cv2.imshow('polyp', frame)
        if SAVE_VIDEO:
            out_video.write(frame)
            
        key = cv2.waitKey(10)
        if key == 27:
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    if SAVE_VIDEO:
        out_video.release()



if __name__ == '__main__':


    if CONVERT:
        print('[INFO] Converting')
        convert()
    else:
        print('[INFO] Infering')
        infer()
