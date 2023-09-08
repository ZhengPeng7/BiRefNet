import os
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import prettytable as pt
import numpy as np

from evaluation.evaluate import evaluator
from config import Config


config = Config()


def do_eval(opt):
    # evaluation for whole dataset
    # dataset first in evaluation
    for _data_name in opt.data_lst:
        pred_data_dir =  sorted(glob(os.path.join(opt.pred_root, opt.model_lst[0], _data_name)))
        if not pred_data_dir:
            print('Skip dataset {}.'.format(_data_name))
            continue
        gt_src = os.path.join(opt.gt_root, _data_name)
        gt_paths = sorted(glob(os.path.join(gt_src, 'gt', '*')))
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(opt.save_dir, '{}_eval.txt'.format(_data_name))
        tb = pt.PrettyTable()
        tb.vertical_char = '&'
        if config.dataset == 'DIS5K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm"]
        elif config.dataset == 'COD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", "maxFm", "meanEm", "maxEm", 'MAE', "adpEm", "adpFm", "HCE"]
        elif config.dataset == 'SOD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE"]
        else:
            tb.field_names = ["Dataset", "Method", "Smeasure", 'MAE', "maxEm", "meanEm", "maxFm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE"]
        for _model_name in opt.model_lst[:]:
            print('\t', 'Evaluating model: {}...'.format(_model_name))
            pred_paths = [p.replace(opt.gt_root, os.path.join(opt.pred_root, _model_name)).replace('/gt/', '/') for p in gt_paths]
            # print(pred_paths[:1], gt_paths[:1])
            em, sm, fm, mae, wfm, hce = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=opt.metrics.split('+'),
                verbose=config.verbose_eval
            )
            if config.dataset == 'DIS5K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                ]
            elif config.dataset == 'COD':
                scores = [
                    sm.round(3), wfm.round(3), fm['curve'].mean().round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), em['curve'].max().round(3), mae.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            elif config.dataset == 'SOD':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mae.round(3),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            else:
                scores = [
                    sm.round(3), mae.round(3), em['curve'].max().round(3), em['curve'].mean().round(3),
                    fm['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                ]
            
            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f' if score <= 1 else '<4').split('.')[-1]
            records = [_data_name, _model_name] + scores
            tb.add_row(records)
            # Write results after every check.
            with open(filename, 'w+') as file_to_write:
                file_to_write.write(str(tb))
        print(tb)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=os.path.join(config.data_root_dir, config.dataset))
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='./e_preds')
    parser.add_argument(
        '--data_lst', type=list, help='test dataset',
        default={
            'DIS5K': ['DIS-VD', 'DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4'][:],
            'COD': ['COD10K', 'NC4K', 'CAMO', 'CHAMELEON'][:],
            'SOD': ['DAVIS-S', 'HRSOD-TE', 'UHRSD-TE', 'DUTS-TE', 'DUT-OMRON'][:]
        }[config.dataset])
    parser.add_argument(
        '--model_lst', type=str, help='candidate competitors',
        default=sorted(glob(os.path.join('ckpt', '*')))[-1])
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='e_results')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=False)
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'HCE'][:100 if config.dataset == 'DIS5K' else -1]))
    opt = parser.parse_args()

    os.makedirs(opt.save_dir, exist_ok=True)
    opt.model_lst = [m for m in sorted(os.listdir(opt.pred_root), key=lambda x: int(x.split('ep')[-1]), reverse=True) if int(m.split('ep')[-1]) % 1 == 0]

    # check the integrity of each candidates
    if opt.check_integrity:
        for _data_name in opt.data_lst:
            for _model_name in opt.model_lst:
                gt_pth = os.path.join(opt.gt_root, _data_name)
                pred_pth = os.path.join(opt.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name, _model_name))
    else:
        print('>>> skip check the integrity of each candidates')

    # start engine
    do_eval(opt)
