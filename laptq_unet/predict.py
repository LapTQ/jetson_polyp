import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torchvision.transforms import ToTensor
from models.unet import UNet

import sys
sys.path.append(sys.path[0] + '/..')

from utils.vid_utils import VideoLoader, ImageLoader


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--half', action='store_true')
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--display', action='store_true')
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--name', type=str, default=None)
    ap.add_argument('--export_video', action='store_true')
    ap.add_argument('--export_image', action='store_true')
    ap.add_argument('--device', type=str, default=None)

    opt = vars(ap.parse_args())

    return opt


def main(opt):

    # process input path
    if opt['input'] == '0':
        opt['input'] = 0
    elif not os.path.exists(opt['input']):
        print('[INFO] Video %s not exists' % opt['input'])
        return

    # process output path
    if opt['output'] is None:
        opt['output'] = str(HERE / '../output')
    if not os.path.isdir(opt['output']):
        os.makedirs(opt['output'], exist_ok=True)
    opt['output'] = Path(opt['output'])

    # process output file name
    assert opt['name'] is None or '.' not in opt['name'], "name must not have extension"
    if opt['name'] is None:
        filename = os.path.basename(str(opt['input']))
        opt['name'], _ = os.path.splitext(filename)
    if opt['half']:
        opt['name'] = opt['name'] + '_half'
        print('[INFO] Halving models')

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

    assert not (opt['export_video'] and opt['export_image']), 'Cannot export image and video at the same time'
    if opt['export_video']:
        H = loader.get_height()
        W = loader.get_width()
        out_video = cv2.VideoWriter(
            str(opt['output'] / (opt['name'] + '.avi')),
            cv2.VideoWriter_fourcc(*'MJPG'),
            FPS,
            (W, H)
        )

    if opt['device'] is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(opt['device'])
    # ================ CHOOSE MODEL ===============
    net = UNet(3, 1)
    # net = deeplabv3_mobilenet_v3_large(pretrained=True)
    # net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    # ================================================
    net.to(device)
    net.load_state_dict(torch.load(opt['weights'], map_location=device))
    if opt['half']:
        net = net.half()

    pbar = tqdm(range(len(loader)))
    for i in pbar:

        # BGR HWC [0, 255] unit8
        ret, img = loader.read()

        condition = not ret or img is None
        if opt['display']:
            condition = condition or cv2.waitKey(int(1000 / FPS)) & 0xFF == ord('q')
        if condition:
            break

        img = cv2.resize(img, (opt['size'], opt['size']))
        # RGB CHW [0., 1.] float32
        x = ToTensor()(img[:, :, ::-1].copy()).unsqueeze(0).to(device)
        if opt['half']:
            x = x.half()

        net.eval()
        with torch.no_grad():
            # ============== CHOOSE MODEL ===============
            y = torch.sigmoid(net(x))
            # y = torch.sigmoid(net(x)['out'])
            # ===========================================

        mask = (y.cpu().squeeze().numpy() * 255).astype('uint8')
        _, mask = cv2.threshold(mask, 0.5 * 255, 255, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

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
    # opt = {
    #     'weights': 'weights/ckpt.pth',
    #     'input': '../../data/well_done/processed_coarse_revise_190420/val/original',
    #     'output': None,
    #     'display': False,
    #     'size': 424,
    #     'export_video': True,
    #     'export_image': False,
    #     'name': None
    # }

    main(opt)
