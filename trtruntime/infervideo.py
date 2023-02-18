

import os
import cv2
import numpy as np
import time
from libs.utilities import scan_dir, create_dir
from libs.common import load_engine, allocate_buffers, TRT_LOGGER, do_inference
from libs.fps import FPS
from infer import TensorRTInfer
from threading import Thread



engine_file_path = '/home/s/model.trt'
video_dir = 'video'
saved_dir = '.'
save_video = False

# create_dir(saved_dir)

video_fps = scan_dir(video_dir)
# print(video_fps)

import pycuda.driver as cuda
import pycuda.autoinit

#engine = load_engine(engine_file_path)
#inputs, outputs, bindings, stream = allocate_buffers(engine)
#context = engine.create_execution_context()
trtEngine = TensorRTInfer(engine_file_path)
cv2.namedWindow('polyp', cv2.WINDOW_NORMAL)


HEIGHT = 384

cap = cv2.VideoCapture(0)#/home/s/Downloads/Colon.mp4 'video/polyp.mp4' 0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) #  1920 1366
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768) # 1080 768

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

if save_video:
    out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (740, 768))
    #out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

skip = False



while True:
    skip = not skip
    if skip:
       continue
    m1 = time.time()
    ret, frame = cap.read()

    if frame is None: 
        break
    #out = np.zeros((780,980*2,3),dtype=np.uint8)
    h,w,_ = frame.shape
    
    frame = frame[:,:740,:]#frame[90:870,200:2*w//3-100,:]#

    test_img = (frame/255.).astype(np.float32)
    test_img = cv2.resize(test_img,(HEIGHT,HEIGHT))
    
    
    cvt_image = np.array([np.transpose(test_img,[2,0,1])])
        #print(cvt_image.shape)
        # Both inputs and outputs are numpy array
        #np.copyto(inputs[0].host, cvt_image.ravel())
        # inputs[0].host = cvt_image
       # [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
    m2 = time.time()
    output = trtEngine.infer(cvt_image)[0]
    m3 = time.time()
#    print(output.shape)    
    output = np.argmax(output, axis=0)
#    print(output.shape)
    
    h, w = output.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[output == 0] = (0, 0, 0)
    mask[output == 1] = (0, 255, 0)
    mask[output == 2] = (0, 0, 255)
    

    res = mask.copy()
    #res[:,:,0] = 0
    res[:,:,1] = cv2.erode(res[:,:,1], kernel, iterations=2)
    res[:,:,1] = cv2.dilate(res[:,:,1], kernel, iterations=2)
    res[:,:,2] = cv2.erode(res[:,:,2], kernel, iterations=2)   # if red, change 0 to 2
    res[:,:,2] = cv2.dilate(res[:,:,2], kernel, iterations=2)   # if red, change 0 to 2
    p = res.copy()
    res[:,:,1] = cv2.dilate(res[:,:,1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    res[:,:,2] = cv2.dilate(res[:,:,2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    bound = res-p
    

    frame_copy = frame.copy()
    print(frame_copy.shape)
    bound = cv2.resize(bound,(frame_copy.shape[1], frame_copy.shape[0]), interpolation = cv2.INTER_NEAREST)
    frame_copy[bound!=0] = bound[bound!=0]
    cv2.imshow("polyp",frame_copy)
    m4 = time.time()
    #print('Read:', m2-m1)
    #print('Infer:', m3-m2)
    #print('Draw:', m4-m3)
    print('FPS:', 1/(m4-m1))
    
    if cv2.waitKey(1) == 27:
        break
    #frame_copy[bound==128] = frame_copy[bound==128]//2+bound[bound==128]
#     out[:,:980,:]=frame
#     out[:,980:,:]=frame_copy
    if save_video:
        out_video.write(frame_copy)

cap.release()
if save_video:
    out_video.release()
    
