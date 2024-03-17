import os
import prettytable as pt

from evaluation.metrics import evaluator
from config import Config


config = Config()

def evaluate(pred_dir, method, testset, only_S_MAE=False, epoch=0):
    filename = os.path.join('evaluation', 'eval-{}.txt'.format(method))
    if os.path.exists(filename):
        id_suffix = 1
        filename = filename.rstrip('.txt') + '_{}.txt'.format(id_suffix)
        while os.path.exists(filename):
            id_suffix += 1
            filename = filename.replace('_{}.txt'.format(id_suffix-1), '_{}.txt'.format(id_suffix))
    gt_paths = sorted([
        os.path.join(config.data_root_dir, config.task, testset, 'gt', p)
        for p in os.listdir(os.path.join(config.data_root_dir, config.task, testset, 'gt'))
    ])
    pred_paths = sorted([os.path.join(pred_dir, method, testset, p) for p in os.listdir(os.path.join(pred_dir, method, testset))])
    with open(filename, 'a+') as file_to_write:
        tb = pt.PrettyTable()
        field_names = [
            "Dataset", "Method", "maxFm", "wFmeasure", 'MAE', "Smeasure", "meanEm", "maxEm", "meanFm",
            "adpEm", "adpFm", 'HCE'
        ]
        tb.field_names = [name for name in field_names if not only_S_MAE or all(metric not in name for metric in ['Em', 'Fm'])]
        em, sm, fm, mae, wfm, hce = evaluator(
            gt_paths=gt_paths[:],
            pred_paths=pred_paths[:],
            metrics=['S', 'MAE', 'E', 'F', 'HCE'][:10*(not only_S_MAE) + 2],    # , 'WF'
            verbose=config.verbose_eval,
        )
        e_max, e_mean, e_adp = em['curve'].max(), em['curve'].mean(), em['adp'].mean()
        f_max, f_mean, f_wfm, f_adp = fm['curve'].max(), fm['curve'].mean(), wfm, fm['adp']
        tb.add_row(
            [
                method+str(epoch), testset, f_max.round(3), f_wfm.round(3), mae.round(3), sm.round(3),
                e_mean.round(3), e_max.round(3), f_mean.round(3), em['adp'].round(3), f_adp.round(3), hce.round(3)
            ] if not only_S_MAE else [method, testset, mae.round(3), sm.round(3)]
        )
        print(tb)
        file_to_write.write(str(tb).replace('+', '|')+'\n')
        file_to_write.close()
    return {'e_max': e_max, 'e_mean': e_mean, 'e_adp': e_adp, 'sm': sm, 'mae': mae, 'f_max': f_max, 'f_mean': f_mean, 'f_wfm': f_wfm, 'f_adp': f_adp, 'hce': hce}


def main():
    only_S_MAE = False
    pred_dir = '.'
    method = 'tmp_val'
    testsets = 'DIS-VD+DIS-TE1'
    for testset in testsets.split('+'):
        res_dct = evaluate(pred_dir, method, testset, only_S_MAE=only_S_MAE)


if __name__ == '__main__':
    main()
