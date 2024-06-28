# --------------------------------------------------------
# Make evaluation along with training. Swith time with space/computation.
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Zheng
# --------------------------------------------------------
import os
from glob import glob
from time import sleep
import argparse
import torch

from config import Config
from models.birefnet import BiRefNet
from dataset import MyData
from evaluation.valid import valid


parser = argparse.ArgumentParser(description='')
parser.add_argument('--cuda_idx', default=-1, type=int)
parser.add_argument('--val_step', default=5*1, type=int)
parser.add_argument('--program_id', default=0, type=int)
# id-th one of this program will evaluate  val_step * N + program_id -th epoch model.
# Test more models, number of programs == number of GPUs: [models[num_all - program_id_1], models[num_all - program_id_max(n, val_step-1)], ...] programs with id>val_step will speed up the evaluation on (val_step - id)%val_step -th epoch models.
# Test fastest, only sequentially searched val_step*N -th models -- set all program_id as the same.
parser.add_argument('--testsets', default='DIS-VD+DIS-TE1+DIS-TE2+DIS-TE3+DIS-TE4', type=str)
args_eval = parser.parse_args()

args_eval.program_id = (args_eval.val_step - args_eval.program_id) % args_eval.val_step

config = Config()
config.only_S_MAE = True
device = 'cpu' if args_eval.cuda_idx < 0 else 'cuda:{}'.format(args_eval.cuda_idx)
ckpt_dir, testsets = glob(os.path.join('ckpt', '*'))[0], args_eval.testsets


def validate_model(model, test_loaders, epoch):
    num_image_testset_all = {'DIS-VD': 470, 'DIS-TE1': 500, 'DIS-TE2': 500, 'DIS-TE3': 500, 'DIS-TE4': 500}
    num_image_testset = {}
    for testset in testsets.split('+'):
        if 'DIS-TE' in testset:
            num_image_testset[testset] = num_image_testset_all[testset]
    weighted_scores = {'f_max': 0, 'sm': 0, 'e_max': 0, 'mae': 0}
    len_all_data_loaders = 0
    model.epoch = epoch
    for testset, data_loader_test in test_loaders.items():
        print('Validating {}...'.format(testset))
        performance_dict = valid(
            model,
            data_loader_test,
            pred_dir='.',
            method=ckpt_dir.split('/')[-1] if ckpt_dir.split('/')[-1].strip('.').strip('/') else 'tmp_val',
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
            for metric in ['sm', 'mae'] if config.only_S_MAE else ['f_max', 'f_wfm', 'sm', 'e_mean', 'mae']:
                weighted_scores[metric] += performance_dict[metric] * len(data_loader_test)
            len_all_data_loaders += len(data_loader_test)
    print('Weighted Scores:')
    for metric, score in weighted_scores.items():
        if score:
            print('\t{}: {:.4f}.'.format(metric, score / len_all_data_loaders))

@torch.no_grad()
def main():
    config = Config()
    # Dataloader
    test_loaders = {}
    for testset in testsets.split('+'):
        dataset = MyData(
            datasets=testset,
            image_size=config.size, is_train=False
        )
        _data_loader_test = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=config.batch_size_valid, num_workers=min(config.num_workers, config.batch_size_valid),
            pin_memory=device != 'cpu', shuffle=False
        )
        print(len(_data_loader_test), "batches of valid dataloader {} have been created.".format(testset))
        test_loaders[testset] = _data_loader_test

    # Model, 3070MiB GPU memory for inference
    model = BiRefNet(bb_pretrained=False).to(device)
    models_evaluated = []
    continous_sleep_time = 0
    while True:
        if (
            (models_evaluated and continous_sleep_time > 60*60*2) or
            (not models_evaluated and continous_sleep_time > 60*60*24)
        ):
            # If no ckpt has been saved, we wait for 24h;
            # elif some ckpts have been saved, we wait for 2h for new ones;
            # else: exit this waiting.
            print('Exiting the waiting for evaluation.')
            break
        models_evaluated_record = 'tmp_models_evaluated.txt'
        if os.path.exists(models_evaluated_record):
            with open(models_evaluated_record, 'r') as f:
                models_evaluated_global = f.read().splitlines()
        else:
            models_evaluated_global = []
        models_detected = [
            m for idx_m, m in enumerate(sorted(
                glob(os.path.join(ckpt_dir, '*.pth')),
                key=lambda x: int(x.rstrip('.pth').split('epoch_')[-1]), reverse=True
            )) if idx_m % args_eval.val_step == args_eval.program_id and m not in models_evaluated + models_evaluated_global
        ]
        if models_detected:
            from time import time
            time_st = time()
            # register the evaluated models
            model_not_evaluated_latest = models_detected[0]
            with open('tmp_models_evaluated.txt', 'a') as f:
                f.write(model_not_evaluated_latest + '\n')
            models_evaluated.append(model_not_evaluated_latest)
            print('Loading {} for validation...'.format(model_not_evaluated_latest))

            # evaluate the current model
            state_dict = torch.load(model_not_evaluated_latest, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            validate_model(model, test_loaders, int(model_not_evaluated_latest.rstrip('.pth').split('epoch_')[-1]))
            continous_sleep_time = 0
            print('Duration of this evaluation:', time() - time_st)
        else:
            sleep_interval = 60 * 2
            sleep(sleep_interval)
            continous_sleep_time += sleep_interval
            continue


if __name__ == '__main__':
    main()
