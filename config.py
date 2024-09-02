import os
import math


class Config():
    def __init__(self) -> None:
              # PATH settings
        # Make up your file system as: SYS_HOME_DIR/codes/dis/BiRefNet, SYS_HOME_DIR/datasets/dis/xx, SYS_HOME_DIR/weights/xx
        if os.name == 'nt': 
            self.sys_home_dir = os.environ['USERPROFILE'] # For windows system
        else:
            self.sys_home_dir = os.environ['HOME'] # For Linux system

        # TASK settings
        self.task = ['DIS5K', 'COD', 'HRSOD', 'General', 'Matting'][0]
        self.training_set = {
            'DIS5K': ['DIS-TR', 'DIS-TR+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4'][0],
            'COD': 'TR-COD10K+TR-CAMO',
            'HRSOD': ['TR-DUTS', 'TR-HRSOD', 'TR-UHRSD', 'TR-DUTS+TR-HRSOD', 'TR-DUTS+TR-UHRSD', 'TR-HRSOD+TR-UHRSD', 'TR-DUTS+TR-HRSOD+TR-UHRSD'][5],
            'General': 'DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4+DIS-TR+TR-HRSOD+TE-HRSOD+TR-HRS10K+TE-HRS10K+TR-UHRSD+TE-UHRSD+TR-P3M-10k+TE-P3M-500-NP+TE-P3M-500-P+TR-humans',    # leave DIS-VD for evaluation.
            'Matting': 'TR-P3M-10k+TE-P3M-500-NP+TR-humans+TR-Distrinctions-646',
        }[self.task]
        self.prompt4loc = ['dense', 'sparse'][0]

        # Faster-Training settings
        self.load_all = False    # Turn it on/off by your case. It may consume a lot of CPU memory. And for multi-GPU (N), it would cost N times the CPU memory to load the data.
        self.use_fp16 = False   # It may cause nan in training.
        self.compile = True and (not self.use_fp16)     # 1. Trigger CPU memory leak in some extend, which is an inherent problem of PyTorch.
                                                        #   Machines with > 70GB CPU memory can run the whole training on DIS5K with default setting.
                                                        # 2. Higher PyTorch version may fix it: https://github.com/pytorch/pytorch/issues/119607.
                                                        # 3. But compile in Pytorch > 2.0.1 seems to bring no acceleration for training.
        self.precisionHigh = True

        # MODEL settings
        self.ms_supervision = True
        self.out_ref = self.ms_supervision and True
        self.dec_ipt = True
        self.dec_ipt_split = True
        self.cxt_num = [0, 3][1]    # multi-scale skip connections from encoder
        self.mul_scl_ipt = ['', 'add', 'cat'][2]
        self.dec_att = ['', 'ASPP', 'ASPPDeformable'][2]
        self.squeeze_block = ['', 'BasicDecBlk_x1', 'ResBlk_x4', 'ASPP_x3', 'ASPPDeformable_x3'][1]
        self.dec_blk = ['BasicDecBlk', 'ResBlk'][0]

        # TRAINING settings
        self.batch_size = 4
        self.finetune_last_epochs = [
            ('IoU', 0),
            {
                'DIS5K': ('IoU', -30),
                'COD': ('IoU', -20),
                'HRSOD': ('IoU', -20),
                'General': ('MAE', -10),
                'Matting': ('MAE', -10),
            }[self.task]
        ][1]    # choose 0 to skip
        self.lr = (1e-4 if 'DIS5K' in self.task else 1e-5) * math.sqrt(self.batch_size / 4)     # DIS needs high lr to converge faster. Adapt the lr linearly
        self.size = 1024
        self.num_workers = max(4, self.batch_size)          # will be decrease to min(it, batch_size) at the initialization of the data_loader

        # Backbone settings
        self.bb = [
            'vgg16', 'vgg16bn', 'resnet50',         # 0, 1, 2
            'swin_v1_t', 'swin_v1_s',               # 3, 4
            'swin_v1_b', 'swin_v1_l',               # 5-bs9, 6-bs4
            'pvt_v2_b0', 'pvt_v2_b1',               # 7, 8
            'pvt_v2_b2', 'pvt_v2_b5',               # 9-bs10, 10-bs5
        ][6]
        self.lateral_channels_in_collection = {
            'vgg16': [512, 256, 128, 64], 'vgg16bn': [512, 256, 128, 64], 'resnet50': [1024, 512, 256, 64],
            'pvt_v2_b2': [512, 320, 128, 64], 'pvt_v2_b5': [512, 320, 128, 64],
            'swin_v1_b': [1024, 512, 256, 128], 'swin_v1_l': [1536, 768, 384, 192],
            'swin_v1_t': [768, 384, 192, 96], 'swin_v1_s': [768, 384, 192, 96],
            'pvt_v2_b0': [256, 160, 64, 32], 'pvt_v2_b1': [512, 320, 128, 64],
        }[self.bb]
        if self.mul_scl_ipt == 'cat':
            self.lateral_channels_in_collection = [channel * 2 for channel in self.lateral_channels_in_collection]
        self.cxt = self.lateral_channels_in_collection[1:][::-1][-self.cxt_num:] if self.cxt_num else []

        # MODEL settings - inactive
        self.lat_blk = ['BasicLatBlk'][0]
        self.dec_channels_inter = ['fixed', 'adap'][0]
        self.refine = ['', 'itself', 'RefUNet', 'Refiner', 'RefinerPVTInChannels4'][0]
        self.progressive_ref = self.refine and True
        self.ender = self.progressive_ref and False
        self.scale = self.progressive_ref and 2
        self.auxiliary_classification = False       # Only for DIS5K, where class labels are saved in `dataset.py`.
        self.refine_iteration = 1
        self.freeze_bb = False
        self.model = [
            'BiRefNet',
        ][0]

        # TRAINING settings - inactive
        self.preproc_methods = ['flip', 'enhance', 'rotate', 'pepper', 'crop'][:4]
        self.optimizer = ['Adam', 'AdamW'][1]
        self.lr_decay_epochs = [1e5]    # Set to negative N to decay the lr in the last N-th epoch.
        self.lr_decay_rate = 0.5
        # Loss
        if self.task not in ['Matting']:
            self.lambdas_pix_last = {
                # not 0 means opening this loss
                # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
                'bce': 30 * 1,          # high performance
                'iou': 0.5 * 1,         # 0 / 255
                'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
                'mae': 30 * 0,
                'mse': 30 * 0,         # can smooth the saliency map
                'triplet': 3 * 0,
                'reg': 100 * 0,
                'ssim': 10 * 1,          # help contours,
                'cnt': 5 * 0,          # help contours
                'structure': 5 * 0,    # structure loss from codes of MVANet. A little improvement on DIS-TE[1,2,3], a bit more decrease on DIS-TE4.
            }
        else:
            self.lambdas_pix_last = {
                # not 0 means opening this loss
                # original rate -- 1 : 30 : 1.5 : 0.2, bce x 30
                'bce': 30 * 0,          # high performance
                'iou': 0.5 * 0,         # 0 / 255
                'iou_patch': 0.5 * 0,   # 0 / 255, win_size = (64, 64)
                'mae': 100 * 1,
                'mse': 30 * 0,         # can smooth the saliency map
                'triplet': 3 * 0,
                'reg': 100 * 0,
                'ssim': 10 * 1,          # help contours,
                'cnt': 5 * 0,          # help contours
                'structure': 5 * 0,    # structure loss from codes of MVANet. A little improvement on DIS-TE[1,2,3], a bit more decrease on DIS-TE4.
            }
        self.lambdas_cls = {
            'ce': 5.0
        }
        # Adv
        self.lambda_adv_g = 10. * 0        # turn to 0 to avoid adv training
        self.lambda_adv_d = 3. * (self.lambda_adv_g > 0)

        # PATH settings - inactive
        self.data_root_dir = os.path.join(self.sys_home_dir, 'datasets/dis')
        self.weights_root_dir = os.path.join(self.sys_home_dir, 'weights')
        self.weights = {
            'pvt_v2_b2': os.path.join(self.weights_root_dir, 'pvt_v2_b2.pth'),
            'pvt_v2_b5': os.path.join(self.weights_root_dir, ['pvt_v2_b5.pth', 'pvt_v2_b5_22k.pth'][0]),
            'swin_v1_b': os.path.join(self.weights_root_dir, ['swin_base_patch4_window12_384_22kto1k.pth', 'swin_base_patch4_window12_384_22k.pth'][0]),
            'swin_v1_l': os.path.join(self.weights_root_dir, ['swin_large_patch4_window12_384_22kto1k.pth', 'swin_large_patch4_window12_384_22k.pth'][0]),
            'swin_v1_t': os.path.join(self.weights_root_dir, ['swin_tiny_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'swin_v1_s': os.path.join(self.weights_root_dir, ['swin_small_patch4_window7_224_22kto1k_finetune.pth'][0]),
            'pvt_v2_b0': os.path.join(self.weights_root_dir, ['pvt_v2_b0.pth'][0]),
            'pvt_v2_b1': os.path.join(self.weights_root_dir, ['pvt_v2_b1.pth'][0]),
        }

        # Callbacks - inactive
        self.verbose_eval = True
        self.only_S_MAE = False
        self.SDPA_enabled = False    # Bugs. Slower and errors occur in multi-GPUs

        # others
        self.device = [0, 'cpu'][0]     # .to(0) == .to('cuda:0')

        self.batch_size_valid = 1
        self.rand_seed = 7
        run_sh_file = [f for f in os.listdir('.') if 'train.sh' == f] + [os.path.join('..', f) for f in os.listdir('..') if 'train.sh' == f]
        if run_sh_file:
            with open(run_sh_file[0], 'r') as f:
                lines = f.readlines()
                self.save_last = int([l.strip() for l in lines if '"{}")'.format(self.task) in l and 'val_last=' in l][0].split('val_last=')[-1].split()[0])

    def print_task(self) -> None:
        # Return task for choosing settings in shell scripts.
        print(self.task)

if __name__ == '__main__':
    config = Config()
    config.print_task()
    
