import argparse
import os

import cv2
import numpy as np
import torch.onnx
import torch

import tensorrt as trt

import sys
sys.path.append(sys.path[0] + '/../..')

from laptq_unet.models.unet import UNet


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--des', type=str, default=None)
    ap.add_argument('--torch2onnx', action='store_true')
    ap.add_argument('--onnx2trt', action='store_true')
    ap.add_argument('--torch2trt', action='store_true')
    ap.add_argument('--batch_size', type=int)    # 8
    ap.add_argument('--input_size', type=int)  # 448
    ap.add_argument('--fp', type=int, required=True)   # 16 32

    opt = ap.parse_args()

    return vars(opt)


def convert_to_onnx(src, des, batch_size, input_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(3, 1).to(device)

    net.load_state_dict(torch.load(src, map_location=device))
    net.eval()

    # input size will be fixed in the exported
    # ONNX graph for all the input’s dimensions,
    # unless specified as a dynamic axes.
    dummy_input = torch.rand(batch_size, 3, input_size, input_size).to(device)
    torch.onnx.export(net, dummy_input, des, input_names=['input'], output_names=['output'], export_params=True)


def convert_to_trt(src, des, precision):
    # os.system(f'trtexec --onnx={src} --saveEngine={des} --fp{precision}')
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(src)

    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass

    # configure builder here
    config = builder.create_builder_config()
    if precision == 16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # profile = builder.create_optimization_profile()
    # profile.set_shape("input", (1, 3, 448, 448), (1, 3, 448, 448), (1, 3, 448, 448))
    # config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)

    with open(des, 'wb') as f:
        f.write(serialized_engine)


def main(opt):

    if opt['des'] is None:
        parent, basename = os.path.split(opt['src'])
        rootname, _ = os.path.splitext(basename)
        if opt['torch2onnx']:
            opt['des'] = os.path.join(parent, rootname + '.onnx')
        elif opt['onnx2trt']:
            opt['des'] = os.path.join(parent, rootname + '.trt')
        elif opt['torch2trt']:
            mid = os.path.join(parent, rootname + '.onnx')
            opt['des'] = os.path.join(parent, rootname + '.trt')

    if opt['torch2onnx']:
        convert_to_onnx(opt['src'], opt['des'], opt['batch_size'], opt['input_size'])
    elif opt['onnx2trt']:
        convert_to_trt(opt['src'], opt['des'], opt['fp'])
    elif opt['torch2trt']:
        convert_to_onnx(opt['src'], mid, opt['batch_size'], opt['input_size'])
        convert_to_trt(mid, opt['des'], opt['fp'])
    print("[INFO] Converted model saved to", opt['des'])


if __name__ == '__main__':
    opt = parse_opt()

    main(opt)

