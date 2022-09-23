import os
import torch


class Config():
    def __init__(self) -> None:
        # Backbone
        self.bb = ['cnn-vgg16', 'cnn-vgg16bn', 'cnn-resnet50', 'trans-pvt'][3]
        self.pvt_weights = ['../bb_weights/pvt_v2_b2.pth', ''][0]
        self.freeze_bb = True

        # Components
        self.dec_blk = ['ResBlk'][0]
        self.dilation = 2
        self.use_bn = self.bb not in ['cnn-vgg16']

        # Data
        self.data_root_dir = '/root/autodl-tmp/datasets/dis'
        self.dataset = 'DIS5K'
        self.size = 1024
        self.batch_size = 15
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'crop', 'pepper'][:1]
        self.num_workers = 8
        self.load_all = False   # 23GB CPU memory to load all sets.
        # On one 3090 + 12 cores Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz, 2.75mins/epoch for training w/ pre-loading vs 7mins/epoch for training w/ online loading.

        # Training
        self.optimizer = ['Adam', 'AdamW'][0]
        self.lr = 1e-4
        self.freeze = True
        self.lr_decay_epochs = [-20]    # Set to negative N to decay the lr in the last N-th epoch.

        # Loss
        self.lambdas_pix_last = {
            # not 0 means opening this loss
            # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
            'bce': 1 * 1,          # high performance
            'iou': 0.05 * 1,         # 0 / 255
            'mse': 150 * 0,         # can smooth the saliency map
            'triplet': 3 * 0,
            'reg': 100 * 0,
            'ssim': 1 * 0,          # help contours
        }
        # Adv
        self.lambda_adv_g = 10. * 0        # turn to 0 to avoid adv training
        self.lambda_adv_d = 3. * (self.lambda_adv_g > 0)

        # others
        self.device = ['cuda', 'cpu'][0]
        self.self_supervision = False
        self.label_smoothing = False

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'go.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'go.sh' == f]
        with open(run_sh_file[0], 'r') as f:
            lines = f.readlines()
            self.val_last = int([l.strip() for l in lines if 'val_last=' in l][0].split('=')[-1])
            self.save_step = int([l.strip() for l in lines if 'step=' in l][0].split('=')[-1])
