import onnx
import onnxruntime as ort
import numpy as np
import cv2
import time

MODEL_PATH = '/home/s/GUI/laptq_unet/tensorrt/repvgg-a1-fulldim.onnx'
HEIGHT = 384
SAVE_VIDEO = False

model = onnx.load(MODEL_PATH)
onnx.checker.check_model(model)

session = ort.InferenceSession(
    MODEL_PATH,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

io_binding = session.io_binding()

cap = cv2.VideoCapture('/home/s/Downloads/Colon.mp4') # '/home/s/Downloads/Colon.mp4'
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024) #  1920 1366
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768) # 1080 768
cv2.namedWindow('polyp', cv2.WINDOW_NORMAL)

if SAVE_VIDEO:
    out_video = cv2.VideoWriter('out_video.avi', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS)/3, (740, 768))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

moving_fps = 25
alpha = 0.25

while True:
    m1 = time.time()
    success, frame = cap.read()

    frame = frame[:,:740,:]

    input_ = frame.copy()
    input_ = cv2.resize(input_, (HEIGHT, HEIGHT)) # resize
    input_ = np.float32(input_ / 255.)    # normalize
    # input_ = input_[:, :, ::-1]   # BRG -> RGB
    input_ = np.moveaxis(input_, 2, 0)    # HWC -> CHW
    input_ = np.expand_dims(input_, axis=0)
    #print(input_.shape)

    m2 = time.time()
    io_binding.bind_cpu_input('input', input_)
    io_binding.bind_output('output')
    session.run_with_iobinding(io_binding)
    output = io_binding.copy_outputs_to_cpu()[0]

    m3 = time.time()
    
    output = np.squeeze(output)
    output = np.argmax(output, axis=0)



    h, w = output.shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[output == 0] = (0, 0, 0)
    mask[output == 1] = (0, 255, 0)
    mask[output == 2] = (0, 0, 255)
    

    res = mask.copy()
    res[:,:,1] = cv2.erode(res[:,:,1], kernel, iterations=2)
    res[:,:,1] = cv2.dilate(res[:,:,1], kernel, iterations=2)
    res[:,:,2] = cv2.erode(res[:,:,2], kernel, iterations=2)   
    res[:,:,2] = cv2.dilate(res[:,:,2], kernel, iterations=2)   
    p = res.copy()
    res[:,:,1] = cv2.dilate(res[:,:,1], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    res[:,:,2] = cv2.dilate(res[:,:,2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations=1)
    bound = res-p
    
    frame_copy = frame.copy()
    bound = cv2.resize(bound,(frame_copy.shape[1], frame_copy.shape[0]), interpolation = cv2.INTER_NEAREST)
    frame_copy[bound!=0] = bound[bound!=0]
    cv2.imshow("polyp",frame_copy)
    m4 = time.time()
    #print('Read:', m2-m1)
    #print('Infer:', m3-m2)
    #print('Draw:', m4-m3)
    moving_fps = alpha * 1/(m4-m1) + (1 - alpha) * moving_fps
    print('FPS:', moving_fps)
    
    if cv2.waitKey(1) == 27:
        break

    if SAVE_VIDEO:
        out_video.write(frame_copy)

cap.release()
cv2.destroyAllWindows()
        
