import os
import math
import psutil
import torch


class Config():
    def __init__(self) -> None:
        self.ms_supervision = True
        self.freeze_bb = 1
        self.load_all = 0
        self.dec_att = ['', 'ASPP'][0]  # Useless for PVTVP
        self.model = ['BSL', 'PVTVP'][0]
        self.IoU_finetune_last_epochs = [-20, 0][0]     # choose 0 to skip
        # Backbone
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'pvt_v2_b2', 'pvt_v2_b5',               # 3-bs10, 4-bs5
            'swin_v1_b', 'swin_v1_l'                # 5-bs9, 6-bs6
        ][3]
        self.weights_root_dir = '/mnt/workspace/workgroup/mohe/weights'
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, 'pvt_v2_b5.pth'),
            'swin_v1_b': os.path.join(self.weights_root_dir, 'swin_base_patch4_window12_384_22kto1k.pth'),
            'swin_v1_l': os.path.join(self.weights_root_dir, 'swin_large_patch4_window12_384_22kto1k.pth'),
        }

        # Components
        self.auxiliary_classification = False
        self.dec_blk = ['BasicBlk', 'BlockA'][0]
        self.dilation = 1   # too slow
        self.dec_channel_inter = ['fixed', 'adap'][0]
        # self.refine = True

        # Training
        self.size = 1024
        self.batch_size = 10
        self.num_workers = min(10, self.batch_size)
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr = 1e-4 * math.sqrt(self.batch_size / 8)  # adapt the lr linearly
        self.lr_decay_epochs = [-10]    # Set to negative N to decay the lr in the last N-th epoch.
        self.only_S_MAE = True

        # Data
        self.data_root_dir = '/mnt/workspace/workgroup/mohe/datasets/dis'
        self.dataset = 'DIS5K'
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:1]

        # Loss
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 30 * 1,          # high performance
            'iou': 0.5 * 1,         # 0 / 255
            'mse': 150 * 0,         # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 5 * 0,          # help contours
        }
        # Adv
        self.lambda_adv_g = 10. * 0        # turn to 0 to avoid adv training
        self.lambda_adv_d = 3. * (self.lambda_adv_g > 0)

        # others
        self.device = ['cuda', 'cpu'][0]

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'go.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'go.sh' == f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
