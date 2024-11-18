import os
import datetime
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from config import Config
from loss import PixLoss, ClsLoss
from dataset import MyData
from models.birefnet import BiRefNet, BiRefNetC2F
from utils import Logger, AverageMeter, set_seed, check_state_dict

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--ckpt_dir', default='ckpt/tmp', help='Temporary folder')
parser.add_argument('--testsets', default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4', type=str)
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
parser.add_argument('--use_accelerate', action='store_true', help='`accelerate launch --multi_gpu train.py --use_accelerate`. Use accelerate for training, good for FP16/BF16/...')
args = parser.parse_args()

if args.use_accelerate:
    from accelerate import Accelerator
    accelerator = Accelerator(
        mixed_precision=['no', 'fp16', 'bf16', 'fp8'][1],
        gradient_accumulation_steps=1,
    )
    args.dist = False

config = Config()
if config.rand_seed:
    set_seed(config.rand_seed)

# DDP
to_be_distributed = args.dist
if to_be_distributed:
    init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*10))
    device = int(os.environ["LOCAL_RANK"])
else:
    device = config.device

epoch_st = 1
# make dir for ckpt
os.makedirs(args.ckpt_dir, exist_ok=True)

# Init log file
logger = Logger(os.path.join(args.ckpt_dir, "log.txt"))
logger_loss_idx = 1

# log model and optimizer params
# logger.info("Model details:"); logger.info(model)
if args.use_accelerate and accelerator.mixed_precision != 'no':
    config.compile = False
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)

if os.path.exists(os.path.join(config.data_root_dir, config.task, args.testsets.strip('+').split('+')[0])):
    args.testsets = args.testsets.strip('+').split('+')
else:
    args.testsets = []


def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
    # Prepare dataloaders
    if to_be_distributed:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size), pin_memory=True,
            shuffle=False, sampler=DistributedSampler(dataset), drop_last=True
        )
    else:
        return torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, num_workers=min(config.num_workers, batch_size, 0), pin_memory=True,
            shuffle=is_train, drop_last=True
        )


def init_data_loaders(to_be_distributed):
    # Prepare datasets
    train_loader = prepare_dataloader(
        MyData(datasets=config.training_set, image_size=config.size, is_train=True),
        config.batch_size, to_be_distributed=to_be_distributed, is_train=True
    )
    print(len(train_loader), "batches of train dataloader {} have been created.".format(config.training_set))
    test_loaders = {}
    for testset in args.testsets:
        _data_loader_test = prepare_dataloader(
            MyData(datasets=testset, image_size=config.size, is_train=False),
            config.batch_size_valid, is_train=False
        )
        print(len(_data_loader_test), "batches of valid dataloader {} have been created.".format(testset))
        test_loaders[testset] = _data_loader_test
    return train_loader, test_loaders


def init_models_optimizers(epochs, to_be_distributed):
    # Init models
    if config.model == 'BiRefNet':
        model = BiRefNet(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    elif config.model == 'BiRefNetC2F':
        model = BiRefNetC2F(bb_pretrained=True and not os.path.isfile(str(args.resume)))
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu', weights_only=True)
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    if not args.use_accelerate:
        if to_be_distributed:
            model = model.to(device)
            model = DDP(model, device_ids=[device])
        else:
            model = model.to(device)
    if config.compile:
        model = torch.compile(model, mode=['default', 'reduce-overhead', 'max-autotune'][0])
    if config.precisionHigh:
        torch.set_float32_matmul_precision('high')

    # Setting optimizer
    if config.optimizer == 'AdamW':
        optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=1e-2)
    elif config.optimizer == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=config.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[lde if lde > 0 else epochs + lde + 1 for lde in config.lr_decay_epochs],
        gamma=config.lr_decay_rate
    )
    logger.info("Optimizer details:"); logger.info(optimizer)
    logger.info("Scheduler details:"); logger.info(lr_scheduler)

    return model, optimizer, lr_scheduler


class Trainer:
    def __init__(
        self, data_loaders, model_opt_lrsch,
    ):
        self.model, self.optimizer, self.lr_scheduler = model_opt_lrsch
        self.train_loader, self.test_loaders = data_loaders
        if args.use_accelerate:
            self.train_loader, self.model, self.optimizer = accelerator.prepare(self.train_loader, self.model, self.optimizer)
            for testset in self.test_loaders.keys():
                self.test_loaders[testset] = accelerator.prepare(self.test_loaders[testset])
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()

    def _train_batch(self, batch):
        if args.use_accelerate:
            inputs = batch[0]#.to(device)
            gts = batch[1]#.to(device)
            class_labels = batch[2]#.to(device)
        else:
            inputs = batch[0].to(device)
            gts = batch[1].to(device)
            class_labels = batch[2].to(device)
        scaled_preds, class_preds_lst = self.model(inputs)
        if config.out_ref:
            (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
            for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True).sigmoid()
                _gdt_label = _gdt_label.sigmoid()
                loss_gdt = self.criterion_gdt(_gdt_pred, _gdt_label) if _idx == 0 else self.criterion_gdt(_gdt_pred, _gdt_label) + loss_gdt
            # self.loss_dict['loss_gdt'] = loss_gdt.item()
        if None in class_preds_lst:
            loss_cls = 0.
        else:
            loss_cls = self.cls_loss(class_preds_lst, class_labels) * 1.0
            self.loss_dict['loss_cls'] = loss_cls.item()

        # Loss
        loss_pix = self.pix_loss(scaled_preds, torch.clamp(gts, 0, 1)) * 1.0
        self.loss_dict['loss_pix'] = loss_pix.item()
        # since there may be several losses for sal, the lambdas for them (lambdas_pix) are inside the loss.py
        loss = loss_pix + loss_cls
        if config.out_ref:
            loss = loss + loss_gdt * 1.0

        self.loss_log.update(loss.item(), inputs.size(0))
        self.optimizer.zero_grad()
        if args.use_accelerate:
            accelerator.backward(loss)
        else:
            loss.backward()
        self.optimizer.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.finetune_last_epochs:
            if config.task == 'Matting':
                self.pix_loss.lambdas_pix_last['mae'] *= 1
                self.pix_loss.lambdas_pix_last['mse'] *= 0.9
                self.pix_loss.lambdas_pix_last['ssim'] *= 0.9
            else:
                self.pix_loss.lambdas_pix_last['bce'] *= 0
                self.pix_loss.lambdas_pix_last['ssim'] *= 1
                self.pix_loss.lambdas_pix_last['iou'] *= 0.5
                self.pix_loss.lambdas_pix_last['mae'] *= 0.9

        for batch_idx, batch in enumerate(self.train_loader):
            self._train_batch(batch)
            # Logger
            if batch_idx % 20 == 0:
                info_progress = 'Epoch[{0}/{1}] Iter[{2}/{3}].'.format(epoch, args.epochs, batch_idx, len(self.train_loader))
                info_loss = 'Training Losses'
                for loss_name, loss_value in self.loss_dict.items():
                    info_loss += ', {}: {:.3f}'.format(loss_name, loss_value)
                logger.info(' '.join((info_progress, info_loss)))
        info_loss = '@==Final== Epoch[{0}/{1}]  Training Loss: {loss.avg:.3f}  '.format(epoch, args.epochs, loss=self.loss_log)
        logger.info(info_loss)

        self.lr_scheduler.step()
        return self.loss_log.avg


def main():

    trainer = Trainer(
        data_loaders=init_data_loaders(to_be_distributed),
        model_opt_lrsch=init_models_optimizers(args.epochs, to_be_distributed)
    )

    for epoch in range(epoch_st, args.epochs+1):
        train_loss = trainer.train_epoch(epoch)
        # Save checkpoint
        # DDP
        if epoch >= args.epochs - config.save_last and epoch % config.save_step == 0:
            torch.save(
                trainer.model.module.state_dict() if to_be_distributed or args.use_accelerate else trainer.model.state_dict(),
                os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            )
    if to_be_distributed:
        destroy_process_group()


if __name__ == '__main__':
    main()
