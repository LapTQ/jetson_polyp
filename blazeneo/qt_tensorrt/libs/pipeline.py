import numpy as np 
import cv2
import sys
import os
import time
import json
from queue import Queue
from threading import Thread, Lock
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # critical, need it though not calling

from .common import load_engine, allocate_buffers, TRT_LOGGER, do_inference
from .fps import FPS


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s\t|%(levelname)s\t|%(funcName)s\t|%(lineno)d\t|%(message)s'
)


class Config():
    def __init__(self, config_file):
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = json.load(f)

        if not self.config['start_app']:
            exit(0)
        
        self.source = self.config["source"]
        self.model_type = self.config["model_type"]
        self.engine_file_path = self.config["engine_file_path"]
        self.engine_type = self.config['engine_type']
        self.width = self.config["width"]
        self.height = self.config["height"]
        self.n_classes = self.config["n_classes"]
        self.threshold = self.config["threshold"]
        self.preprocess_mode = self.config["preprocess_mode"]
        self.postprocess_mode = self.config["postprocess_mode"]
        self.class_labels = self.config['class_labels']
        self.x_start = self.config["crop_coordinates"]["x_start"]
        self.y_start = self.config["crop_coordinates"]["y_start"]
        self.x_end = self.config["crop_coordinates"]["x_end"]
        self.y_end = self.config["crop_coordinates"]["y_end"]

        self.camera_width = self.config["camera_width"]
        self.camera_height = self.config["camera_height"]

        self.stopped = False

    def is_stop(self):
        return self.stopped

    def stop(self):
        self.stopped = True


class Streamming:
    """
    Class that continously get frames from stream with a dedicated thread. 
    """
    def __init__(self, config):

        self.cfg = config

        self.capture = cv2.VideoCapture(self.cfg.source)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera_height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera_width)

    def read(self):

        if self.cfg.is_stop():
            self.stop()
            return False, None

        (ret, frame) = self.capture.read()
        # if ret:
        #     frame = cv2.resize(frame, (self.cfg.camera_width, self.cfg.camera_height))

        return ret, frame

    
    def stop(self):
        self.capture.release()

    def is_opened(self):
        return self.capture.isOpened()


class Inferencing:
    """
    Create TensorRT model and infer image in seperate thread
    """
    def __init__(self, preprocessing, postprocessing, config):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

        self.cfg = config

        self.moving_fps = 25
        self.moving_sec = 0
        self.alpha = 0.25

        if self.cfg.engine_type == 'onnx':
            self._load_onnx_model()
        elif self.cfg.engine_type == 'trt':
            self._load_trt_model()
        elif self.cfg.engine_type == 'tftrt':
            if self.cfg.model_type == 'detection':
                self._load_detection_tftrt_fp16_model()
            
            

        logging.info('Model loaded')

    
    def process(self, image):
        if self.cfg.engine_type == 'onnx':
            return self._process_onnx(image)
        elif self.cfg.engine_type == 'trt':
            return self._process_trt(image)
        elif self.cfg.engine_type == 'tftrt':
            if self.cfg.model_type == 'detection':
                return self._process_detection_tftrt16(image)
            
    
    def _load_trt_model(self):
        
        
        class TensorRTInfer:

            def __init__(self, engine_path):
                # Load TRT engine
                self.logger = trt.Logger(trt.Logger.ERROR)
                with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                self.context = self.engine.create_execution_context()
                assert self.engine
                assert self.context

                # Setpup I/O bindings
                self.inputs = []
                self.outputs = []
                self.allocations = []
                for i in range(self.engine.num_bindings):
                    is_input = False
                    if self.engine.binding_is_input(i):
                        is_input = True
                    name = self.engine.get_binding_name(i)
                    dtype = self.engine.get_binding_dtype(i)
                    shape = self.engine.get_binding_shape(i)
                    if is_input:
                        self.batch_size = shape[0]
                    size = np.dtype(trt.nptype(dtype)).itemsize
                    for s in shape:
                        size *= s
                    allocation = cuda.mem_alloc(size)
                    binding = {
                        'index': i,
                        'name': name,
                        'dtype': np.dtype(trt.nptype(dtype)),
                        'shape': list(shape),
                        'allocation': allocation,
                    }
                    self.allocations.append(allocation)
                    if self.engine.binding_is_input(i):
                        self.inputs.append(binding)
                    else:
                        self.outputs.append(binding)
                
                assert self.batch_size > 0
                assert len(self.inputs) > 0
                assert len(self.outputs) > 0
                assert len(self.allocations) > 0

            def input_spec(self):
                """
                Get the specs for the input tensor of the networl. Useful to prepare memory allocations.
                :return: Two items, the shape of the input tensor and its (numpy) datatype.
                """
                return self.inputs[0]['shape'], self.inputs[0]['dtype']

            def output_spec(self):
                """
                Get the specs for the output tensor of the network. Useful to prepare memory allocations.
                :return: Two items, the shape of the output tensor and its (numpy) datatype.
                """
                return self.outputs[0]['shape'], self.outputs[0]['dtype']

            def infer(self, batch):
                """
                Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
                :param batch: A numpy array holding the image batch.
                """
                # Prepare the output data
                output = np.zeros(*self.output_spec())

                # Process I/O and execute the network
                cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
                self.context.execute_v2(self.allocations)
                cuda.memcpy_dtoh(output, self.outputs[0]['allocation'])

                return output
        
        self.trtEngine = TensorRTInfer(self.cfg.engine_file_path)


    
    def _load_onnx_model(self):
        import onnx
        import onnxruntime as ort

        logging.info(f'Loading segmentation onnx model at {self.cfg.engine_file_path}')

        model = onnx.load(self.cfg.engine_file_path)
        onnx.checker.check_model(model)

        self.session = ort.InferenceSession(
            self.cfg.engine_file_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        self.io_binding = self.session.io_binding()

    
    def _load_detection_tftrt_fp16_model(self):
        import tensorflow as tf

        # np_config.enable_numpy_behavior()
        logging.info(f'Loading detection tftrt model at {self.cfg.engine_file_path}')
        logging.debug(tf.config.list_physical_devices('GPU'))

        model = tf.saved_model.load(self.cfg.engine_file_path)
        self.func = model.signatures['serving_default']
        self.output_tensorname = list(self.func.structured_outputs.keys())[0]

    
    def _process_trt(self, image):

        start = time.time()
        crop_image = image[self.cfg.y_start:self.cfg.y_end, self.cfg.x_start:self.cfg.x_end]
        logging.debug(f'image.shape == {image.shape}')
        logging.debug(f'crop_image.shape == {crop_image.shape}')

        input_ = self.preprocessing(crop_image)
        # input_ = np.expand_dims(input_, axis=0)

        output = self.trtEngine.infer(input_)[0]
        output = np.squeeze(output)

        image = self.postprocessing(image, crop_image, output)

        end = time.time()
        self.moving_fps = self.alpha * 1/(end - start) + (1 - self.alpha) * self.moving_fps
        self.moving_sec = self.alpha * (end - start) + (1 - self.alpha) * self.moving_sec
        logging.debug("{}: Inference time of {:.4f}s with {:.1f} fps".format('Inferencing', self.moving_sec, self.moving_fps))

        return image

    def _process_onnx(self, image):

        start = time.time()
        crop_image = image[self.cfg.y_start:self.cfg.y_end, self.cfg.x_start:self.cfg.x_end]
        logging.debug(f'image.shape == {image.shape}')
        logging.debug(f'crop_image.shape == {crop_image.shape}')

        input_ = self.preprocessing(crop_image)
        input_ = np.expand_dims(input_, axis=0)

        self.io_binding.bind_cpu_input(self.session.get_inputs()[0].name, input_)
        self.io_binding.bind_output(self.session.get_outputs()[0].name)
        self.session.run_with_iobinding(self.io_binding)
        output = self.io_binding.copy_outputs_to_cpu()[0]
        output = np.squeeze(output)

        image = self.postprocessing(image, crop_image, output)

        end = time.time()
        self.moving_fps = self.alpha * 1/(end - start) + (1 - self.alpha) * self.moving_fps
        self.moving_sec = self.alpha * (end - start) + (1 - self.alpha) * self.moving_sec
        logging.debug("{}: Inference time of {:.4f}s with {:.1f} fps".format('Inferencing', self.moving_sec, self.moving_fps))

        return image

    
    def _process_detection_tftrt16(self, image):
        import tensorflow as tf
        start = time.time()
        
        crop_image = image[self.cfg.y_start:self.cfg.y_end, self.cfg.x_start:self.cfg.x_end]
        input_ = self.preprocessing(crop_image)
        input_ = tf.expand_dims(input_, axis=0)
        output = self.func(input_)[self.output_tensorname]
        output = np.squeeze(output)

        image = self.postprocessing(image, crop_image, output)

        end = time.time()
        self.moving_fps = self.alpha * 1/(end - start) + (1 - self.alpha) * self.moving_fps
        self.moving_sec = self.alpha * (end - start) + (1 - self.alpha) * self.moving_sec
        logging.debug("{}: Inference time of {:.4f}s with {:.1f} fps".format('Inferencing', self.moving_sec, self.moving_fps))

        return image



class Displaying:
    """
    Class that continously get frames with output from output queue with a dedicated thread.
    """
    def __init__(self, config, name="display"):

        self.window_name = name

        self.cfg = config

        self.save_video = False
        self.__setupSaveVideo()

        self.__setupWindow()


    def display(self, image):
        
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1)
            
        if key == ord('q') or key == 27:
            self.stop()
        elif key == ord(' '):
            cv2.waitKey(0)
        
        if self.save_video:
            self.out_video.write(image)


    def __setupWindow(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def stop(self):
        self.cfg.stop()
        cv2.destroyAllWindows()
        if self.save_video:
            self.out_video.release()

    def __setupSaveVideo(self):
        if self.save_video:
            logging.info('Saving video')
            self.out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30.0, (self.cfg.camera_width, self.cfg.camera_height))