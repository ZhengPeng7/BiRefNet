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
from models.birefnet import BiRefNet
from utils import Logger, AverageMeter, set_seed, check_state_dict
from evaluation.valid import valid

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank
from torch.cuda import amp


parser = argparse.ArgumentParser(description='')
parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
parser.add_argument('--epochs', default=120, type=int)
parser.add_argument('--trainset', default='DIS5K', type=str, help="Options: 'DIS5K'")
parser.add_argument('--ckpt_dir', default=None, help='Temporary folder')
parser.add_argument('--testsets', default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4', type=str)
parser.add_argument('--dist', default=False, type=lambda x: x == 'True')
args = parser.parse_args()


config = Config()
if config.rand_seed:
    set_seed(config.rand_seed)

if config.use_fp16:
    # Half Precision
    scaler = amp.GradScaler(enabled=config.use_fp16)

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
logger.info("datasets: load_all={}, compile={}.".format(config.load_all, config.compile))
logger.info("Other hyperparameters:"); logger.info(args)
print('batch size:', config.batch_size)


if os.path.exists(os.path.join(config.data_root_dir, config.task, args.testsets.strip('+').split('+')[0])):
    args.testsets = args.testsets.strip('+').split('+')
else:
    args.testsets = []

# Init model
def prepare_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, to_be_distributed=False, is_train=True):
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
    # Prepare dataset
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
    model = BiRefNet(bb_pretrained=True)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume, map_location='cpu')
            state_dict = check_state_dict(state_dict)
            model.load_state_dict(state_dict)
            global epoch_st
            epoch_st = int(args.resume.rstrip('.pth').split('epoch_')[-1]) + 1
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
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
        if config.out_ref:
            self.criterion_gdt = nn.BCELoss() if not config.use_fp16 else nn.BCEWithLogitsLoss()

        # Setting Losses
        self.pix_loss = PixLoss()
        self.cls_loss = ClsLoss()
        
        # Others
        self.loss_log = AverageMeter()
        if config.lambda_adv_g:
            self.optimizer_d, self.lr_scheduler_d, self.disc, self.adv_criterion = self._load_adv_components()
            self.disc_update_for_odd = 0

    def _load_adv_components(self):
        # AIL
        from loss import Discriminator
        disc = Discriminator(channels=3, img_size=config.size)
        if to_be_distributed:
            disc = disc.to(device)
            disc = DDP(disc, device_ids=[device], broadcast_buffers=False)
        else:
            disc = disc.to(device)
        if config.compile:
            disc = torch.compile(disc, mode=['default', 'reduce-overhead', 'max-autotune'][0])
        adv_criterion = nn.BCELoss() if not config.use_fp16 else nn.BCEWithLogitsLoss()
        if config.optimizer == 'AdamW':
            optimizer_d = optim.AdamW(params=disc.parameters(), lr=config.lr, weight_decay=1e-2)
        elif config.optimizer == 'Adam':
            optimizer_d = optim.Adam(params=disc.parameters(), lr=config.lr, weight_decay=0)
        lr_scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d,
            milestones=[lde if lde > 0 else args.epochs + lde + 1 for lde in config.lr_decay_epochs],
            gamma=config.lr_decay_rate
        )
        return optimizer_d, lr_scheduler_d, disc, adv_criterion

    def _train_batch(self, batch):
        inputs = batch[0].to(device)
        gts = batch[1].to(device)
        class_labels = batch[2].to(device)
        if config.use_fp16:
            with amp.autocast(enabled=config.use_fp16):
                scaled_preds, class_preds_lst = self.model(inputs)
                if config.out_ref:
                    (outs_gdt_pred, outs_gdt_label), scaled_preds = scaled_preds
                    for _idx, (_gdt_pred, _gdt_label) in enumerate(zip(outs_gdt_pred, outs_gdt_label)):
                        _gdt_pred = nn.functional.interpolate(_gdt_pred, size=_gdt_label.shape[2:], mode='bilinear', align_corners=True)#.sigmoid()
                        # _gdt_label = _gdt_label.sigmoid()
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

                if config.lambda_adv_g:
                    # gen
                    valid = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                    adv_loss_g = self.adv_criterion(self.disc(scaled_preds[-1] * inputs), valid) * config.lambda_adv_g
                    loss += adv_loss_g
                    self.loss_dict['loss_adv'] = adv_loss_g.item()
                    self.disc_update_for_odd += 1
            # self.loss_log.update(loss.item(), inputs.size(0))
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            if config.lambda_adv_g and self.disc_update_for_odd % 2 == 0:
                # disc
                fake = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(0.0), requires_grad=False).to(device)
                adv_loss_real = self.adv_criterion(self.disc(gts * inputs), valid)
                adv_loss_fake = self.adv_criterion(self.disc(scaled_preds[-1].detach() * inputs.detach()), fake)
                adv_loss_d = (adv_loss_real + adv_loss_fake) / 2 * config.lambda_adv_d
                self.loss_dict['loss_adv_d'] = adv_loss_d.item()
                # self.optimizer_d.zero_grad()
                # adv_loss_d.backward()
                # self.optimizer_d.step()
                self.optimizer_d.zero_grad()
                scaler.scale(adv_loss_d).backward()
                scaler.step(self.optimizer_d)
                scaler.update()
        else:
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

            if config.lambda_adv_g:
                # gen
                valid = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(1.0), requires_grad=False).to(device)
                adv_loss_g = self.adv_criterion(self.disc(scaled_preds[-1] * inputs), valid) * config.lambda_adv_g
                loss += adv_loss_g
                self.loss_dict['loss_adv'] = adv_loss_g.item()
                self.disc_update_for_odd += 1
            self.loss_log.update(loss.item(), inputs.size(0))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if config.lambda_adv_g and self.disc_update_for_odd % 2 == 0:
                # disc
                fake = Variable(torch.cuda.FloatTensor(scaled_preds[-1].shape[0], 1).fill_(0.0), requires_grad=False).to(device)
                adv_loss_real = self.adv_criterion(self.disc(gts * inputs), valid)
                adv_loss_fake = self.adv_criterion(self.disc(scaled_preds[-1].detach() * inputs.detach()), fake)
                adv_loss_d = (adv_loss_real + adv_loss_fake) / 2 * config.lambda_adv_d
                self.loss_dict['loss_adv_d'] = adv_loss_d.item()
                self.optimizer_d.zero_grad()
                adv_loss_d.backward()
                self.optimizer_d.step()

    def train_epoch(self, epoch):
        global logger_loss_idx
        self.model.train()
        self.loss_dict = {}
        if epoch > args.epochs + config.IoU_finetune_last_epochs:
            self.pix_loss.lambdas_pix_last['bce'] *= 0
            self.pix_loss.lambdas_pix_last['ssim'] *= 1
            self.pix_loss.lambdas_pix_last['iou'] *= 0.5

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
        if config.lambda_adv_g:
            self.lr_scheduler_d.step()
        return self.loss_log.avg

    def validate_model(self, epoch):
        num_image_testset_all = {'DIS-VD': 470, 'DIS-TE1': 500, 'DIS-TE2': 500, 'DIS-TE3': 500, 'DIS-TE4': 500}
        num_image_testset = {}
        for testset in args.testsets:
            if 'DIS-TE' in testset:
                num_image_testset[testset] = num_image_testset_all[testset]
        weighted_scores = {'f_max': 0, 'f_mean': 0, 'f_wfm': 0, 'sm': 0, 'e_max': 0, 'e_mean': 0, 'mae': 0}
        len_all_data_loaders = 0
        self.model.epoch = epoch
        for testset, data_loader_test in self.test_loaders.items():
            print('Validating {}...'.format(testset))
            performance_dict = valid(
                self.model,
                data_loader_test,
                pred_dir='.',
                method=args.ckpt_dir.split('/')[-1] if args.ckpt_dir.split('/')[-1].strip('.').strip('/') else 'tmp_val',
                testset=testset,
                only_S_MAE=config.only_S_MAE,
                device=device
            )
            print('Test set: {}:'.format(testset))
            if config.only_S_MAE:
                print('Smeasure: {:.4f}, MAE: {:.4f}'.format(
                    performance_dict['sm'], performance_dict['mae']
                ))
            else:
                print('Fmax: {:.4f}, Fwfm: {:.4f}, Smeasure: {:.4f}, Emean: {:.4f}, MAE: {:.4f}'.format(
                    performance_dict['f_max'], performance_dict['f_wfm'], performance_dict['sm'], performance_dict['e_mean'], performance_dict['mae']
                ))
            if '-TE' in testset:
                for metric in ['sm', 'mae'] if config.only_S_MAE else ['f_max', 'f_mean', 'f_wfm', 'sm', 'e_max', 'e_mean', 'mae']:
                    weighted_scores[metric] += performance_dict[metric] * len(data_loader_test)
                len_all_data_loaders += len(data_loader_test)
        print('Weighted Scores:')
        for metric, score in weighted_scores.items():
            if score:
                print('\t{}: {:.4f}.'.format(metric, score / len_all_data_loaders))


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
                trainer.model.module.state_dict() if to_be_distributed else trainer.model.state_dict(),
                os.path.join(args.ckpt_dir, 'epoch_{}.pth'.format(epoch))
            )
        if config.val_step and epoch >= args.epochs - config.save_last and (args.epochs - epoch) % config.val_step == 0:
            if to_be_distributed:
                if get_rank() == 0:
                    print('Validating at rank-{}...'.format(get_rank()))
                    trainer.validate_model(epoch)
            else:
                trainer.validate_model(epoch)
    if to_be_distributed:
        destroy_process_group()

if __name__ == '__main__':
    main()
