import os
import math
import psutil
import torch


class Config():
    def __init__(self) -> None:
        self.load_all = 0#psutil.virtual_memory().free // (2 ** (10 * 3)) > 23 + 8
        self.freeze_bb = 1
        self.model = ['BSL', 'PVTVP'][1]
        self.dec_att = ['', 'ASPP'][0]  # Useless for PVTVP
        # Backbone
        self.bb = ['cnn-vgg16', 'cnn-vgg16bn', 'cnn-resnet50', 'trans-pvt'][3]
        self.pvt_weights = ['/mnt/workspace/workgroup/mohe/weights/pvt_v2_b2.pth', ''][0]

        # Components
        self.auxiliary_classification = False
        self.dec_blk = ['ResBlk'][0]
        self.dilation = 1   # too slow
        self.dec_channel_inter = ['fixed', 'adap'][0]
        self.use_bn = self.bb not in ['cnn-vgg16']
        # self.refine = True

        # Data
        self.data_root_dir = '/mnt/workspace/workgroup/mohe/datasets/dis'
        self.dataset = 'DIS5K'
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:1]
        # Check free CPU mem. 23GB CPU memory to load all sets, save 3 mins for each epoch. 8G mem for main pipeline.
        # Training
        self.size = 512
        # See current free GPU memory to set the batch_size for training
        free_mem_gpu = torch.cuda.mem_get_info()[0] / (2 ** (10 * 3))
        self.batch_size = 10    #int((free_mem_gpu - (5 + 2 * (not self.freeze_bb))) / (3.5 + 4.5 * (not self.freeze_bb)))   # (free_mem_gpu - base model mem) // per batch mem
        self.num_workers = min(10, self.batch_size)
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr = 1e-4 * math.sqrt(self.batch_size / 8)  # adapt the lr linearly
        self.lr_decay_epochs = [-10]    # Set to negative N to decay the lr in the last N-th epoch.
        self.only_S_MAE = False

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
        self.label_smoothing = False

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'go.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'go.sh' == f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
