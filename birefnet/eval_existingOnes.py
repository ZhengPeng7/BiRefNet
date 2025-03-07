import os
import argparse
from glob import glob
import prettytable as pt

from evaluation.metrics import evaluator
from config import Config


config = Config()


def do_eval(args):
    # evaluation for whole dataset
    # dataset first in evaluation
    for _data_name in args.data_lst.split('+'):
        pred_data_dir =  sorted(glob(os.path.join(args.pred_root, args.model_lst[0], _data_name)))
        if not pred_data_dir:
            print('Skip dataset {}.'.format(_data_name))
            continue
        gt_src = os.path.join(args.gt_root, _data_name)
        gt_paths = sorted(glob(os.path.join(gt_src, 'gt', '*')))
        print('#' * 20, _data_name, '#' * 20)
        filename = os.path.join(args.save_dir, '{}_eval.txt'.format(_data_name))
        tb = pt.PrettyTable()
        tb.vertical_char = '&'
        if config.task == 'DIS5K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU']
        elif config.task == 'COD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", "meanEm", "maxEm", 'MAE', "maxFm", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU']
        elif config.task == 'HRSOD':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU']
        elif config.task == 'General':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU']
        elif config.task == 'General-2K':
            tb.field_names = ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU']
        elif config.task == 'Matting':
            tb.field_names = ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MSE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU']
        else:
            tb.field_names = ["Dataset", "Method", "Smeasure", 'MAE', "maxEm", "meanEm", "maxFm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU']
        for _model_name in args.model_lst[:]:
            print('\t', 'Evaluating model: {}...'.format(_model_name))
            pred_paths = [p.replace(args.gt_root, os.path.join(args.pred_root, _model_name)).replace('/gt/', '/') for p in gt_paths]
            # print(pred_paths[:1], gt_paths[:1])
            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=args.metrics.split('+'),
                verbose=config.verbose_eval
            )
            if config.task == 'DIS5K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            elif config.task == 'COD':
                scores = [
                    sm.round(3), wfm.round(3), fm['curve'].mean().round(3), em['curve'].mean().round(3), em['curve'].max().round(3), mae.round(3),
                    fm['curve'].max().round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            elif config.task == 'HRSOD':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mae.round(3),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            elif config.task == 'General':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            elif config.task == 'General-2K':
                scores = [
                    fm['curve'].max().round(3), wfm.round(3), mae.round(3), sm.round(3), em['curve'].mean().round(3), int(hce.round()), 
                    em['curve'].max().round(3), fm['curve'].mean().round(3), em['adp'].round(3), fm['adp'].round(3),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            elif config.task == 'Matting':
                scores = [
                    sm.round(3), fm['curve'].max().round(3), em['curve'].mean().round(3), mse.round(5),
                    em['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3), em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]
            else:
                scores = [
                    sm.round(3), mae.round(3), em['curve'].max().round(3), em['curve'].mean().round(3),
                    fm['curve'].max().round(3), fm['curve'].mean().round(3), wfm.round(3),
                    em['adp'].round(3), fm['adp'].round(3), int(hce.round()),
                    mba.round(3), biou['curve'].max().round(3), biou['curve'].mean().round(3),
                ]

            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1  else format(score, '<4')
            records = [_data_name, _model_name] + scores
            tb.add_row(records)
            # Write results after every check.
            with open(filename, 'w+') as file_to_write:
                file_to_write.write(str(tb)+'\n')
        print(tb)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gt_root', type=str, help='ground-truth root',
        default=os.path.join(config.data_root_dir, config.task))
    parser.add_argument(
        '--pred_root', type=str, help='prediction root',
        default='./e_preds')
    parser.add_argument(
        '--data_lst', type=str, help='test dataset',
        default=config.testsets.replace(',', '+'))
    parser.add_argument(
        '--save_dir', type=str, help='candidate competitors',
        default='e_results')
    parser.add_argument(
        '--check_integrity', type=bool, help='whether to check the file integrity',
        default=False)
    parser.add_argument(
        '--metrics', type=str, help='candidate competitors',
        default='+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE'][:100 if 'DIS5K' in config.task else -1]))
    args = parser.parse_args()
    args.metrics = '+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE'][:100 if sum(['DIS-' in _data for _data in args.data_lst.split('+')]) else -1])

    os.makedirs(args.save_dir, exist_ok=True)
    try:
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root), key=lambda x: int(x.split('epoch_')[-1]), reverse=True) if int(m.split('epoch_')[-1]) % 1 == 0]
    except:
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root))]

    # check the integrity of each candidates
    if args.check_integrity:
        for _data_name in args.data_lst.split('+'):
            for _model_name in args.model_lst:
                gt_pth = os.path.join(args.gt_root, _data_name)
                pred_pth = os.path.join(args.pred_root, _model_name, _data_name)
                if not sorted(os.listdir(gt_pth)) == sorted(os.listdir(pred_pth)):
                    print(len(sorted(os.listdir(gt_pth))), len(sorted(os.listdir(pred_pth))))
                    print('The {} Dataset of {} Model is not matching to the ground-truth'.format(_data_name, _model_name))
    else:
        print('>>> skip check the integrity of each candidates')

    # start engine
    do_eval(args)
