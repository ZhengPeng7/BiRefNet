import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from config import Config


class Discriminator(nn.Module):
    def __init__(self, channels=1, img_size=256):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class IoU_loss(torch.nn.Module):
    def __init__(self):
        super(IoU_loss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU


class ThrReg_loss(torch.nn.Module):
    def __init__(self):
        super(ThrReg_loss, self).__init__()

    def forward(self, pred, gt=None):
        return torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))


class PixLoss(nn.Module):
    """
    IoU loss for outputs in [1:] scales.
    """
    def __init__(self):
        super(PixLoss, self).__init__()
        self.config = Config()
        self.lambdas_pix_last = self.config.lambdas_pix_last

        self.criterions_last = {}
        if 'bce' in self.lambdas_pix_last and self.lambdas_pix_last['bce']:
            self.criterions_last['bce'] = nn.BCELoss()
        if 'iou' in self.lambdas_pix_last and self.lambdas_pix_last['iou']:
            self.criterions_last['iou'] = IoU_loss()
        if 'ssim' in self.lambdas_pix_last and self.lambdas_pix_last['ssim']:
            self.criterions_last['ssim'] = SSIMLoss()
        if 'mse' in self.lambdas_pix_last and self.lambdas_pix_last['mse']:
            self.criterions_last['mse'] = nn.MSELoss()
        if 'reg' in self.lambdas_pix_last and self.lambdas_pix_last['reg']:
            self.criterions_last['reg'] = ThrReg_loss()


    def forward(self, scaled_preds, gt):
        loss = 0
        for _, pred_lvl in enumerate(scaled_preds):
            if pred_lvl.shape != gt.shape:
                pred_lvl = nn.functional.interpolate(pred_lvl, size=gt.shape[2:], mode='bilinear', align_corners=True)
            pred_lvl = pred_lvl.sigmoid()
            for criterion_name, criterion in self.criterions_last.items():
                loss += criterion(pred_lvl, gt) * self.lambdas_pix_last[criterion_name]
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1, 1)(x)
    mu_y = nn.AvgPool2d(3, 1, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)


def saliency_structure_consistency(x, y):
    ssim = torch.mean(SSIM(x,y))
    return ssim
