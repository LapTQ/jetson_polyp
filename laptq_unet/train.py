import argparse
import os
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

import albumentations as A
import cv2

from models.unet import UNet
from utils.dataset import get_dataloader


HERE = Path(__file__).parent


def parse_opt():

    ap = argparse.ArgumentParser()

    ap.add_argument('--epoch', type=int, default=40)
    ap.add_argument('--train_img', type=str)
    ap.add_argument('--train_lbl', type=str)
    ap.add_argument('--dev_img', type=str)
    ap.add_argument('--dev_lbl', type=str)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-1)
    ap.add_argument('--weights', type=str, default=None)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--focal_alpha', type=float, default=-1)
    ap.add_argument('--focal_gamma', type=float, default=2)
    ap.add_argument('--earlystop_patience', type=int, default=None, help='monitoring dev loss')

    opt = vars(ap.parse_args())

    return opt


import torch
from torch.nn import functional as F


def main(opt):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ================ CHOOSE MODEL ===============
    net = UNet(3, 1)
    # net = deeplabv3_mobilenet_v3_large(pretrained=True)
    # net.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=1)
    # ================================================
    net.to(device)

    if opt['weights']:
        print('Loading pretrained at ' + opt['weights'])
        net.load_state_dict(torch.load(opt['weights'], map_location=device))

    # ================ CHOOSE MODEL ===============
    out_size = net(torch.zeros((2, 3, opt['size'], opt['size']), dtype=torch.float32).to(device)).shape[-1]
    # out_size = net(torch.zeros((2, 3, opt['size'], opt['size']), dtype=torch.float32).to(device))['out'].shape[-1]
    # =============================================

    # ================= AUGMENTATION ==================== # TODO check
    transform = A.Compose([
        A.ISONoise(p=0.5),
        A.MotionBlur(blur_limit=(3, 6), p=0.5),
        A.SafeRotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(p=1),
    ])
    # ===================================================

    train_loader = get_dataloader(
        img_dir=opt['train_img'],
        lbl_dir=opt['train_lbl'],
        batch_size=opt['batch_size'],
        in_size=opt['size'],
        out_size=out_size,
        transform=transform,
        shuffle=True
    )

    dev_loader = get_dataloader(
        img_dir=opt['dev_img'],
        lbl_dir=opt['dev_lbl'],
        batch_size=opt['batch_size'],
        in_size=opt['size'],
        out_size=out_size,
        transform=None,
        shuffle=False
    )

    # ============== TRAINING CONFIGURATION ===============
    criterion = sigmoid_focal_loss # nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt['lr'], momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=4, verbose=True)
    weight_name = datetime.now().strftime("%Y%m%d%H%M%S") + '.pth' # f'ckpt_{len(os.listdir(HERE/"weights"))}.pth'
    min_total_dev_loss = 1e9    # for my model checkpoint callback
    earlystop_streak = 0
    # =====================================================

    for epoch in range(opt['epoch']):

        with tqdm(enumerate(train_loader), ascii=True, desc=f'Epoch {epoch + 1} [{len(train_loader)}]', unit=' batch') as t:
            net.train()
            running_loss = 0.0
            total_loss = 0.0
            for i, data in t:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                # =============== CHOOSE MODEL ================
                outputs = net(inputs)
                # outputs = net(inputs)['out']
                # =============================================
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()
                running_loss += float(loss)
                total_loss += float(loss)
                if i % 3 == 2:
                    t.set_postfix(loss=running_loss/3.0)
                    running_loss = 0.0

            total_loss /= len(train_loader)

            # evaluation
            net.eval()
            total_dev_loss = 0.0
            for data in dev_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # =============== CHOOSE MODEL ================
                outputs = net(inputs)
                # outputs = net(inputs)['out']
                # =============================================
                loss = criterion(outputs, labels)
                total_dev_loss += float(loss)

            total_dev_loss /= len(dev_loader)
            # t.set_postfix(loss=total_loss, dev_loss=total_dev_loss, lr=optimizer.param_groups[0]['lr'])
            print('lr=', optimizer.param_groups[0]['lr'], ', total_loss=', total_loss, ', total_dev_loss=', total_dev_loss)

            # ================== schedulers =================
            # reduce on plateau
            scheduler.step(total_loss)

            if total_dev_loss + 1e-4 < min_total_dev_loss:
                # model checkpoint
                print(f'[INFO] dev loss improved from {min_total_dev_loss:.6f} to {total_dev_loss:.6f}')
                min_total_dev_loss = total_dev_loss
                if not os.path.isdir('weights'):
                    os.mkdir('weights')
                torch.save(net.state_dict(), str(HERE / ('weights/' + weight_name)))
                print('[CALLBACK] weights saved at:', str(HERE / ('weights/' + weight_name)))
                
                # early stopping
                earlystop_streak = 0
            else:
                print(f'[INFO] dev loss did NOT improved from {min_total_dev_loss:.6f}')
                
                # early stopping
                earlystop_streak += 1
            
            # early stop
            
            if opt['earlystop_patience'] and earlystop_streak >= opt['earlystop_patience']:
                print('[CALLBACK] Early stopped, reach maximun patience:',  opt['earlystop_patience'])
                break
                
                
            




if __name__ == '__main__':

    opt = parse_opt()


    def sigmoid_focal_loss(
            inputs: torch.Tensor,
            targets: torch.Tensor,
            alpha: float = opt['focal_alpha'],  # 0.75 0.25
            gamma: float = opt['focal_gamma'],  # 3 2
            reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Args:
            targets: 0 for the negative class and 1 for the positive class
            alpha: (optional) Weight in range (0,1) to balance positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    main(opt)

    # python polyp/laptq_unet/train.py --train_img data/Neo-final/images --train_lbl data/Neo-final/mask_images --dev_img data/well_done/processed_result_test_190420/val/original --dev_lbl data/well_done/processed_result_test_190420/val/segmentation/ --batch_size 4 --lr 1e-1 --size 424 --epoch 40 --focal_gamma 2.0



# BCELoss: model with sigmoid, target float
# BCEWithLogitsLoss: model without sigmoid, target float?
# CrossEntropyLoss: model without softmax, target can be index (int) or onehot (float?)
