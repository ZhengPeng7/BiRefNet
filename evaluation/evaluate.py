import os
import cv2
import argparse
import prettytable as pt

from evaluation.metrics import evaluator
from config import Config


config = Config()

def evaluate(pred_dir, method, testset, only_S_MAE=False, epoch=0):
    filename = os.path.join('evaluation', 'eval-{}.txt'.format(method))
    gt_paths = [
        os.path.join(config.data_root_dir, config.dataset, testset, 'gt', p)
        for p in os.listdir(os.path.join(config.data_root_dir, config.dataset, testset, 'gt'))
    ]
    pred_paths = [os.path.join(pred_dir, method, testset, p) for p in os.listdir(os.path.join(pred_dir, method, testset))]
    with open(filename, 'a+') as file_to_write:
        tb = pt.PrettyTable()
        field_names = [
            "Dataset", "Method", "maxEm", "Smeasure", "maxFm", "MAE", "meanEm", "meanFm",
            "adpEm", "wFmeasure", "adpFm"
        ]
        tb.field_names = [name for name in field_names if not only_S_MAE or all(metric not in name for metric in ['Em', 'Fm'])]
        em, sm, fm, mae, wfm = evaluator(
            gt_paths=gt_paths[:],
            pred_paths=pred_paths[:],
            metrics=['S', 'MAE', 'E', 'F', 'WF'][:10*(not only_S_MAE) + 2]
        )
        tb.add_row(
            [
                method+str(epoch), testset, em['curve'].max().round(3), sm.round(3), fm['curve'].max().round(3), mae.round(3), em['curve'].mean().round(3), fm['curve'].mean().round(3),
                em['adp'].round(3), wfm.round(3), fm['adp'].round(3)
            ] if not only_S_MAE else [method, testset, sm.round(3), mae.round(3)]
        )
        print(tb)
        file_to_write.write(str(tb)+'\n')
        file_to_write.close()
    return {'em': em, 'sm': sm, 'fm': fm, 'mae': mae, 'wfm': wfm}


def main():
    only_S_MAE = False
    pred_dir = '.'
    method = 'tmp_val'
    testsets = 'DIS-VD+DIS-TE1'
    for testset in testsets.split('+'):
        evaluate(pred_dir, method, testset, only_S_MAE=only_S_MAE)


if __name__ == '__main__':
    main()
