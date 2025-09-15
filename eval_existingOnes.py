import os
import argparse
from glob import glob
import prettytable as pt

from evaluation.metrics import evaluator, sort_and_round_scores
from config import Config


config = Config()


def do_eval(args):
    task_to_field_names = {
        'DIS5K': ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU'],
        'COD': ["Dataset", "Method", "Smeasure", "wFmeasure", "meanFm", "meanEm", "maxEm", 'MAE', "maxFm", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU'],
        'HRSOD': ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MAE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU'],
        'General': ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU'],
        'Matting': ["Dataset", "Method", "Smeasure", "maxFm", "meanEm", 'MSE', "maxEm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU'],
        'General-2K': ["Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "HCE", "maxEm", "meanFm", "adpEm", "adpFm", 'mBA', 'maxBIoU', 'meanBIoU'],
        'Others': ["Dataset", "Method", "Smeasure", 'MAE', "maxEm", "meanEm", "maxFm", "meanFm", "wFmeasure", "adpEm", "adpFm", "HCE", 'mBA', 'maxBIoU', 'meanBIoU'],
    }
    for data_name in args.data_lst.split('+'):
        print('#' * 20, data_name, '#' * 20)
        if not glob(os.path.join(args.pred_root, args.model_lst[0], data_name)):
            print('Skip dataset {}.'.format(data_name))
            continue
        gt_paths = sorted(glob(os.path.join(args.gt_root, data_name, 'gt', '*')))

        tb = pt.PrettyTable()
        tb.vertical_char = '&'
        tb.field_names = task_to_field_names[config.task] if config.task in task_to_field_names else task_to_field_names['Others']
        for model_name in args.model_lst[:]:
            print('\t', 'Evaluating model: {}...'.format(model_name))
            pred_paths = [p.replace(args.gt_root, os.path.join(args.pred_root, model_name)).replace('/gt/', '/') for p in gt_paths]

            em, sm, fm, mae, mse, wfm, hce, mba, biou = evaluator(
                gt_paths=gt_paths,
                pred_paths=pred_paths,
                metrics=args.metrics.split('+'),
                verbose=config.verbose_eval,
                num_workers=8,
            )
            scores = sort_and_round_scores(config.task, [em, sm, fm, mae, mse, wfm, hce, mba, biou])
            for idx_score, score in enumerate(scores):
                scores[idx_score] = '.' + format(score, '.3f').split('.')[-1] if score <= 1  else format(score, '<4')
            records = [data_name, model_name] + scores
            tb.add_row(records)
            os.makedirs(args.save_dir, exist_ok=True)
            with open(os.path.join(args.save_dir, '{}_eval.txt'.format(data_name)), 'w+') as file_to_write:
                file_to_write.write(str(tb)+'\n')
        print(tb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_root', type=str, help='ground-truth root', default=os.path.join(config.data_root_dir, config.task))
    parser.add_argument('--pred_root', type=str, help='prediction root', default='./e_preds')
    parser.add_argument('--data_lst', type=str, help='test datasets', default=config.testsets.replace(',', '+'))
    parser.add_argument('--save_dir', type=str, help='directory to save results', default='e_results')
    parser.add_argument('--metrics', type=str, help='candidate competitors', default='+'.join(['S', 'MAE']))
    args = parser.parse_args()

    if args.metrics == 'all':
        args.metrics = '+'.join(['S', 'MAE', 'E', 'F', 'WF', 'MBA', 'BIoU', 'MSE', 'HCE'][:100 if sum(['DIS-' in _data for _data in args.data_lst.split('+')]) else -1])

    try:
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root), key=lambda x: int(x.split('epoch_')[-1].split('-')[0]), reverse=True) if int(m.split('epoch_')[-1].split('-')[0]) % 1 == 0]
    except Exception as e:
        print(f"Exception: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")
        args.model_lst = [m for m in sorted(os.listdir(args.pred_root))]

    do_eval(args)
