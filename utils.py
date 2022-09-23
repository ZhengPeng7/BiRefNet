import logging
import os
import torch
from torchvision import transforms
import numpy as np
import random
import cv2


def generate_smoothed_gt(gts):
    epsilon = 0.001
    new_gts = (1-epsilon)*gts+epsilon/2
    return new_gts


class Logger():
    def __init__(self, path="log.txt"):
        self.logger = logging.getLogger('BSL')
        self.file_handler = logging.FileHandler(path, "w")
        self.stdout_handler = logging.StreamHandler()
        self.stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stdout_handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
    
    def info(self, txt):
        self.logger.info(txt)
    
    def close(self):
        self.file_handler.close()
        self.stdout_handler.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, path, filename="latest.pth"):
    torch.save(state, os.path.join(path, filename))


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


def save_tensor_merge(tenor_im, tensor_mask, path, colormap='HOT'):
    im = tenor_im.cpu().detach().clone()
    im = im.squeeze(0).numpy()
    im = ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-20)) * 255
    im = np.array(im,np.uint8)
    mask = tensor_mask.cpu().detach().clone()
    mask = mask.squeeze(0).numpy()
    mask = ((mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-20)) * 255
    mask = np.clip(mask, 0, 255)
    mask = np.array(mask, np.uint8)
    if colormap == 'HOT':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_HOT)
    elif colormap == 'PINK':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_PINK)
    elif colormap == 'BONE':
        mask = cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_BONE)
    # exec('cv2.applyColorMap(mask[0,:,:], cv2.COLORMAP_' + colormap+')')
    im = im.transpose((1, 2, 0))
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    mix = cv2.addWeighted(im, 0.3, mask, 0.7, 0)
    cv2.imwrite(path, mix)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
