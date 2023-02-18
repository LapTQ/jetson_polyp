import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

import sys
sys.path.append(sys.path[0] + '/../..')

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from laptq_unet.utils.vid_utils import VideoLoader, ImageLoader



HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--engine', type=str, required=True)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--export_video', action='store_true')
    ap.add_argument('--export_image', action='store_true')


    opt = ap.parse_args()

    return vars(opt)


def main(opt):

    # process input path
    if opt['input'] == '0':
        opt['input'] = 0
    elif not os.path.exists(opt['input']):
        print('[INFO] Video %s not exists' % opt['input'])
        return

    # process output path
    if opt['output'] is None:
        opt['output'] = str(HERE / '../../output')
    if not os.path.isdir(opt['output']):
        os.makedirs(opt['output'], exist_ok=True)
    opt['output'] = Path(opt['output'])

    # process output file name
    assert opt['name'] is None or '.' not in opt['name'], "name must not have extension"
    if opt['name'] is None:
        filename = os.path.basename(str(opt['input']))
        opt['name'], _ = os.path.splitext(filename)
        opt['name'] = opt['name'] + '_trt'

    # create loader (image/video)
    if os.path.isdir(opt['input']):
        loader = ImageLoader.Builder(opt['input'], is_dir=True).get_product()
    else:
        if os.path.splitext(os.path.basename(str(opt['input'])))[1].lower() in ['.png', 'jpg', '.jpeg']:
            loader = ImageLoader.Builder(opt['input'], is_dir=False).get_product()
        else:
            loader = VideoLoader.Builder(opt['input']).get_product()

    FPS = loader.get_fps()

    if opt['display']:
        cv2.namedWindow(opt['name'], cv2.WINDOW_NORMAL)

    H = loader.get_height()
    W = loader.get_width()

    assert not (opt['export_video'] and opt['export_image']), 'Cannot export image and video at the same time'
    if opt['export_video']:
        out_video = cv2.VideoWriter(
            str(opt['output'] / (opt['name'] + '.avi')),
            cv2.VideoWriter_fourcc(*'MJPG'),
            FPS,
            (W, H)
        )

    with open(opt['engine'], 'rb') as f:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

    for binding in engine:
        if engine.binding_is_input(binding):    # expecting only 1 input
            input_shape = engine.get_binding_shape(binding)
            input_dtype = trt.nptype(engine.get_binding_dtype(binding))
            input_size = np.empty(input_shape, dtype=input_dtype).nbytes
        else:   # and 1 output
            output_shape = engine.get_binding_shape(binding)
            output_dtype = trt.nptype(engine.get_binding_dtype(binding))
            output_size = np.empty(output_shape, dtype=output_dtype).nbytes

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)
    
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=output_dtype)
    
    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    pbar = tqdm(range(len(loader)))
    for i in pbar:

        # BGR HWC [0, 255] unit8
        ret, img = loader.read()

        condition = not ret or img is None
        if opt['display']:
            condition = condition or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q')
        if condition:
            break

        img = cv2.resize(img, (input_shape[3], input_shape[2]))
        # BGR HWC [0, 255] to RGB CHW [0., 1.]
        h_input = np.moveaxis(img[:, :, ::-1] / 255., 2, 0).reshape(input_shape).astype(input_dtype).copy()

        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings, stream.handle, None)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        
        y = torch.sigmoid(torch.Tensor(h_output).reshape(tuple(output_shape)))# .astype('float32')

        mask = (y.squeeze().numpy() * 255).astype('uint8')

        _, mask = cv2.threshold(mask, 0.5 * 255, 255, cv2.THRESH_BINARY)

        cnts, hie = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        show_img = cv2.drawContours(img, cnts, -1, (0, 255, 0), thickness=2)
        show_img = cv2.resize(show_img, (W, H))

        if opt['display']:
            cv2.imshow(opt['name'], show_img)

        if opt['export_video']:
            out_video.write(show_img)

        if opt['export_image']:
            if len(loader) > 1:
                out_img_name = opt['name'] + '.png'
            else:
                out_img_name = opt['name'] + '_' + str(i) + '.png'
            cv2.imwrite(str(opt.output / out_img_name), show_img)

    loader.release()

    if opt['display']:
        cv2.destroyAllWindows()

    if opt['export_video']:
        out_video.release()
        print('[INFO] Video saved in', str(opt['output'] / (opt['name'] + '.avi')))



if __name__ == '__main__':

    opt = parse_opt()
    main(opt)



# https://github.com/NVIDIA/TensorRT/blob/main/quickstart/SemanticSegmentation/tutorial-runtime.ipynb
# https://github.com/NVIDIA/TensorRT/blob/main/quickstart/IntroNotebooks/4.%20Using%20PyTorch%20through%20ONNX.ipynb
