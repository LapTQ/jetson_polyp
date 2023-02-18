import os
import cv2
import numpy as np
import time
from libs.utilities import scan_dir, create_dir
from libs.common import load_engine, allocate_buffers, TRT_LOGGER, do_inference
from libs.fps import FPS
from infer import TensorRTInfer
engine_file_path = 'neofinal_b1.trt'
video_dir = 'video'
saved_dir = '.'

# create_dir(saved_dir)

video_fps = scan_dir(video_dir)
# print(video_fps)

import pycuda.driver as cuda
import pycuda.autoinit

#engine = load_engine(engine_file_path)
#inputs, outputs, bindings, stream = allocate_buffers(engine)
#context = engine.create_execution_context()
trtEngine = TensorRTInfer(engine_file_path)
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
# cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
# cv2.setWindowProperty('test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_fps = [0]
for video_path in video_fps:
    print(video_path)
    #fn = video_path.split('/')[-1]
    #print(">>>>> process video: ", fn)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    
    # out = cv2.VideoWriter(os.path.join(saved_dir, fn), cv2.VideoWriter_fourcc(*'VP80'), 60, (320*2, 320))

    fps = FPS().start()
    while(cap.isOpened()):
        start = time.time()
        ret, frame = cap.read()
        if ret == False:
            break
        
        m1 = time.time()
        
        h, w, c = frame.shape
        image = frame[:, :int(w*0.6777), :]

        image = cv2.resize(image, (384, 384))
        #cvt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cvt_image = image.astype(np.float32)
        cvt_image = cvt_image/255.
        
        cvt_image = np.array([np.transpose(cvt_image,[2,0,1])])
        #print(cvt_image.shape)
        # Both inputs and outputs are numpy array
        #np.copyto(inputs[0].host, cvt_image.ravel())
        # inputs[0].host = cvt_image
       # [output] = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=1)
        m2 = time.time()
        output = trtEngine.infer(cvt_image)[0]
        m3 = time.time()
        
        #print(output.shape)
        #print("infer",time.time()-infer_start)
        output = np.argmax(output, axis=0)
        h, w = output.shape[:2]
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask[output == 0] = (0, 0, 0)
        mask[output == 1] = (0, 255, 0)
        mask[output == 2] = (0, 0, 255)
        pr_mask = mask#(output)*255

        # convert mask from 1 channel to 3 channel
        #pr_mask = np.uint8(pr_mask)
       # print(pr_mask.shape)
        #pr_mask = np.reshape(pr_mask, (384, 384))
        #pr_mask = np.stack((pr_mask,)*3, axis=-1)

        tmp = np.zeros([96, 96*2, 3],dtype=np.uint8)
        tmp[:, :96, :] = image
        tmp[:, 96:, :] = pr_mask
        m4 = time.time()

        cv2.imshow('test', tmp)
        fps.update()
        end = time.time()
        print("FPS", 1/(end-start))
        print("Read: %.3f \t Preprocess: %.3f \t Infer: %.3f \t Postprocess: %.3f \t Show: %.3f" % ((m1 - start), (m2 - m1), (m3 - m2), (m4 - m3), (end - m4)))
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # out.write(tmp)

    cap.release()
    # out.release()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

